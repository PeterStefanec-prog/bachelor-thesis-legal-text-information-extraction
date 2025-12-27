import re, math, textwrap
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

from sentence_transformers import SentenceTransformer
import faiss
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# ----------------------------
# 1) Chunking (page -> chunks)
# ----------------------------

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ \t]+\n", "\n\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()

def split_into_sentences(text: str):
    protected = {
        "sp. zn.": "sp_zn",
        "č.": "c_",
        "JUDr.": "JUDr_",
        "Mgr.": "Mgr_",
        "Ing.": "Ing_",
        "PhD.": "PhD_",
        "s.r.o.": "sro_",
        "a.s.": "as_"
    }
    for k,v in protected.items():
        text = text.replace(k, v)

    parts = re.split(r"(?<=[\.\?\!])\s+", text)

    out = []
    for s in parts:
        for k,v in protected.items():
            s = s.replace(v, k)
        s = s.strip()
        if s:
            out.append(s)
    return out

def chunk_sentences(sentences, target_chars=900, min_chars=350):
    chunks=[]
    cur=[]
    cur_len=0
    for s in sentences:
        if cur_len + len(s) + 1 > target_chars and cur_len >= min_chars:
            chunks.append(" ".join(cur).strip())
            cur=[s]
            cur_len=len(s)
        else:
            cur.append(s)
            cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

def build_block_chunks(pdf_path: Path, min_chars=120, max_chars=2200):
    d = fitz.open(str(pdf_path))
    chunks = []

    for pno in range(len(d)):
        page = d[pno]
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)

        for bidx, b in enumerate(blocks):
            x0, y0, x1, y1, text, *_rest = b

            # block_type niekedy býva v _rest; bezpečne filtruj len textové bloky:
            if not isinstance(text, str):
                continue

            t = text.replace("\u00a0", " ")
            t = re.sub(r"-\n", "", t)
            t = re.sub(r"\s+", " ", t).strip()

            if len(t) < min_chars:
                continue

            # (voliteľné) ak je blok extrémne dlhý, nechaj ho – kvôli highlightu
            # alebo ho rozdeľ, ale bbox potom bude stále ten istý (teda highlightne väčší kus)
            if len(t) > max_chars:
                # môžeš buď ponechať, alebo rozsekať text na subchunk-y
                # tu nechám ponechať, aby highlight sedel čo najlepšie
                pass

            chunks.append({
                "global_id": len(chunks),
                "page": pno + 1,
                "chunk_on_page": bidx,
                "bbox": [x0, y0, x1, y1],   # dôležité!
                "text": t
            })

    d.close()
    return chunks




# ----------------------------
# 2) BM25
# ----------------------------

TOKEN_RE = re.compile(r"[a-záäčďéíľĺňóôŕšťúýž0-9§]+", re.IGNORECASE)

def tokenize_sk(text: str):
    return TOKEN_RE.findall(text.lower())

def bm25_build(tokenized_docs, k1=1.5, b=0.75):
    N = len(tokenized_docs)
    doc_lens = np.array([len(d) for d in tokenized_docs], dtype=float)
    avgdl = doc_lens.mean() if N else 0.0

    df = Counter()
    for d in tokenized_docs:
        df.update(set(d))

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))

    tfs = [Counter(d) for d in tokenized_docs]
    return {"doc_lens": doc_lens, "avgdl": avgdl, "idf": idf, "tfs": tfs, "k1": k1, "b": b}

def bm25_score(query_tokens, bm25_obj):
    idf = bm25_obj["idf"]
    tfs = bm25_obj["tfs"]
    k1 = bm25_obj["k1"]
    b = bm25_obj["b"]
    doc_lens = bm25_obj["doc_lens"]
    avgdl = bm25_obj["avgdl"]

    scores = np.zeros(len(tfs), dtype=float)
    for i, tf in enumerate(tfs):
        dl = doc_lens[i]
        denom_const = k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else k1
        s = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + denom_const)
        scores[i] = s
    return scores

def bm25_top(query: str, tokenized_docs, bm25_obj, top_n=30):
    q_tokens = tokenize_sk(query)
    scores = bm25_score(q_tokens, bm25_obj)
    idx = np.argsort(-scores)[:top_n]
    return idx.tolist()


# ----------------------------
# 3) Dense embeddings (E5) + FAISS
# ----------------------------

def build_faiss_index(embeddings: np.ndarray):
    # cosine similarity = inner product on L2-normalized vectors
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def dense_top(model: SentenceTransformer, index, chunks_text, query: str, top_n=30):
    # E5 style: prefix queries with "query: " and documents with "passage: "
    q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
    q_emb = q_emb.astype("float32")
    scores, ids = index.search(q_emb, top_n)
    return ids[0].tolist()


# ----------------------------
# 4) RRF fusion + channels
# ----------------------------

def rrf_fuse(lists, k=60):
    score = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst, start=1):
            score[idx] += 1.0 / (k + rank)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [i for i,_ in ranked]

def build_channel_rank(queries, tokenized_docs, bm25_obj, model, faiss_index, chunks_text, top_n=30, rrf_k=60):
    score = defaultdict(float)
    for q in queries:
        bm = bm25_top(q, tokenized_docs, bm25_obj, top_n=top_n)
        de = dense_top(model, faiss_index, chunks_text, q, top_n=top_n)
        fused = rrf_fuse([bm, de], k=rrf_k)
        for rank, idx in enumerate(fused, start=1):
            score[idx] += 1.0 / (rrf_k + rank)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ----------------------------
# 5) Highlight in PDF + report
# ----------------------------

# def pick_search_phrase(text, max_words=10):
#     words = re.findall(r"\w+|§", text)
#     phrase = " ".join(words[:max_words])
#     return phrase if len(phrase) >= 20 else " ".join(words[:6])

def pick_phrases(text: str):
    words = re.findall(r"\w+|§", text)
    anchors = []

    # 3 kotvy: začiatok, stred, koniec
    starts = [0, max(0, len(words)//2 - 5), max(0, len(words) - 10)]
    for start in starts:
        phrase = " ".join(words[start:start+10])
        if len(phrase) >= 20:
            anchors.append(phrase)

    # zahoď duplicity (poradie zachovaj)
    seen = set()
    uniq = []
    for a in anchors:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq

def highlight_chunks(pdf_in: Path, selected_chunks, pdf_out: Path):
    d = fitz.open(str(pdf_in))

    for ch in selected_chunks:
        page = d[ch["page"] - 1]
        x0, y0, x1, y1 = ch["bbox"]
        chunk_rect = fitz.Rect(x0, y0, x1, y1)

        annot = page.add_highlight_annot(chunk_rect)
        annot.update()

    d.save(str(pdf_out), garbage=4, deflate=True)
    d.close()


# def highlight_chunks(pdf_in: Path, selected_chunks, pdf_out: Path):
#     d = fitz.open(str(pdf_in))
#
#     for ch in selected_chunks:
#         page = d[ch["page"] - 1]
#         x0, y0, x1, y1 = ch["bbox"]
#         chunk_rect = fitz.Rect(x0, y0, x1, y1)
#
#         # zober všetky slová na stránke
#         words = page.get_text("words")
#
#         rects = []
#         for w in words:
#             wrect = fitz.Rect(w[0], w[1], w[2], w[3])
#             if wrect.intersects(chunk_rect):
#                 rects.append(wrect)
#
#         if rects:
#             annot = page.add_highlight_annot(rects)
#             annot.update()
#         else:
#             print(f"[WARN] No words inside bbox for chunk global_id={ch['global_id']} page={ch['page']}")
#
#     d.save(str(pdf_out), garbage=4, deflate=True)
#     d.close()


def make_report(selected_chunks, out_path: Path):
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    x = 2*cm
    y = height - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Selected evidence passages (top 20)")
    y -= 1*cm

    for ch in selected_chunks:
        header = f"Page {ch['page']} | chunk_on_page {ch['chunk_on_page']} | global_id {ch['global_id']}"
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, header)
        y -= 0.5*cm

        c.setFont("Helvetica", 9)
        wrapped = textwrap.wrap(ch["text"], width=110)
        for line in wrapped:
            if y < 2*cm:
                c.showPage()
                y = height - 2*cm
                c.setFont("Helvetica", 9)
            c.drawString(x, y, line)
            y -= 0.4*cm
        y -= 0.4*cm

        if y < 2*cm:
            c.showPage()
            y = height - 2*cm

    c.save()


# ----------------------------
# Main
# ----------------------------

PENALTY_QUERIES = [
    "zmluvná pokuta",
    "neprimeraná zmluvná pokuta",
    "primeraná zmluvná pokuta",
    "zníženie zmluvnej pokuty",
    "moderačné oprávnenie zmluvná pokuta",
    "určitosť dojednania o zmluvnej pokute",
    "neurčitosť dojednania o zmluvnej pokute",
    "§ 544 zmluvná pokuta",
    "§ 301 zmluvná pokuta"
]

CONTEXT_QUERIES = [
    "uzavreli zmluvu",
    "predmet zmluvy",
    "zmluvné strany žalobca žalovaný",
    "skutkový stav",
    "porušenie povinnosti",
    "odstúpenie od zmluvy",
    "cena plnenia faktúra",
    "omeškanie zaplatenie"
]


def main():
    pdf_path = Path("Rozsudok_14Cob-85-2021.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path.resolve()}")

    print("1) Building chunks...")
    chunks = build_block_chunks(pdf_path)
    chunks_text = [c["text"] for c in chunks]
    print(f"   chunks: {len(chunks)}")

    print("2) Building BM25...")
    tokenized = [tokenize_sk(t) for t in chunks_text]
    bm25_obj = bm25_build(tokenized)

    print("3) Loading embedding model (E5)...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    print("4) Encoding chunk embeddings...")
    # E5 expects passages prefixed with "passage: "
    embs = model.encode([f"passage: {t}" for t in chunks_text],
                        batch_size=16,
                        normalize_embeddings=True,
                        show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    print("5) Building FAISS index...")
    index = build_faiss_index(embs)

    print("6) Ranking channels (BM25 + dense + RRF)...")
    pen_ranked = build_channel_rank(PENALTY_QUERIES, tokenized, bm25_obj, model, index, chunks_text)
    ctx_ranked = build_channel_rank(CONTEXT_QUERIES, tokenized, bm25_obj, model, index, chunks_text)

    topK_penalty = [i for i,_ in pen_ranked[:12]]
    topK_context = [i for i,_ in ctx_ranked[:8]]

    # must-consider heuristics
    must_pen = set(i for i,t in enumerate(chunks_text) if ("zmluv" in t.lower() and "pokut" in t.lower()))
    must_ctx = set(i for i,t in enumerate(chunks_text) if ("zmluva" in t.lower() and ("predmet" in t.lower() or "dielo" in t.lower())))

    selected = set(topK_penalty) | set(topK_context) | must_pen | must_ctx

    # add neighbors
    for i in list(selected):
        if i-1 >= 0: selected.add(i-1)
        if i+1 < len(chunks_text): selected.add(i+1)

    # final pick: 20
    pen_score = {i:s for i,s in pen_ranked}
    ctx_score = {i:s for i,s in ctx_ranked}
    final_scored = [(i, pen_score.get(i,0.0)+ctx_score.get(i,0.0)) for i in selected]
    final_scored.sort(key=lambda x: x[1], reverse=True)
    final_ids = [i for i,_ in final_scored[:20]]

    selected_chunks = [chunks[i] for i in final_ids]

    # save table
    rows=[]
    for rank,i in enumerate(final_ids, start=1):
        c = chunks[i]
        rows.append({
            "rank": rank,
            "global_id": i,
            "page": c["page"],
            "chunk_on_page": c["chunk_on_page"],
            "len_chars": len(c["text"]),
            "preview": c["text"][:220] + ("…" if len(c["text"])>220 else "")
        })
    df = pd.DataFrame(rows)
    df.to_csv("selected_chunks.csv", index=False)
    print("   Wrote selected_chunks.csv")

    # annotate pdf
    out_pdf = Path("annotated_selected.pdf")
    print("7) Highlighting in PDF...")
    highlight_chunks(pdf_path, selected_chunks, out_pdf)
    print(f"   Wrote {out_pdf}")

    # report
    out_rep = Path("selected_passages_report.pdf")
    print("8) Writing report PDF...")
    make_report(selected_chunks, out_rep)
    print(f"   Wrote {out_rep}")

    print("Done.")

if __name__ == "__main__":
    main()
