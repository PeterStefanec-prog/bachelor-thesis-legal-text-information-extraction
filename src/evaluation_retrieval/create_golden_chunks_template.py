import os
import json
import csv

# --- 1. SETUP PATHS ---
# hah must run this from the root folder
INPUT_DIR = "data/02_processed_json"
EVAL_DIR = "data/05_evaluation"
TEXTS_DIR = os.path.join(EVAL_DIR, "texts_for_annotation")

# Create folders if they don't exist
os.makedirs(TEXTS_DIR, exist_ok=True)

# --- 2. MY SELECTED DOCUMENTS ---
# Here I manually put the 10 files I want to use for my Golden Dataset.
# (Just copy-paste the filenames from data/02_processed_json)
SELECTED_DOCS = [
    # "KS_Bratislava_1CoZm_36_2016_00_dokument.json",
    # "KS_Bratislava_3Cob_464_2012_00_dokument.json",
    # "NS_SR_5Obdo_14_2023_00_dokument.json",
    # TODO: various little bit random docs - decisions
    "NS_SR_1Cdo_85_2023_00_dokument.json",
    "KS_Bratislava_1Cob_40_2018_00_dokument.json",  # pokuta primerana
    "KS_Bratislava_1Cob_130_2019_00_dokument.json", # primerana
    "NS_SR_1Obdo_66_2018_00_dokument.json", # excelentny pripad
    "NS_SR_1Obdo_72_2019_00_dokument.json", # excelentny pripad - pokuta znizena
    "KS_Bratislava_2Co_380_2011_00_dokument.json",
    "KS_Trenčín_8Cob_64_2011_00_dokument.json",
    "KS_Trenčín_8Cob_258_2014_00_dokument.json",
    "KS_Trnava_32Cob_3_2021_00_dokument.json", # realne moderuje
    "KS_Banská_Bystrica_43CoPv_10_2023_00_dokument.json"




]
# KS_Bratislava_1Cob_47_2022_00_dokument - very bad one - paragraf 301 je spomenuty len na zaciatku z predoslych rozhodntui
# KS_Bratislava_1Cob_93_2016_00_dokument - also bad one - paragraphs 300 - 302 are mentioned only as citations of contract
# KS_Bratislava_1Cob_117_2020_00_dokument - also bad one - paragraphs 300 - 302 are mentioned only as citations of contract
# KS_Bratislava_1Cob_187_2015_00_dokument - nedostal sa k ucinnosti

def main():
    csv_data = []

    print("Startin template generation...")

    for doc_name in SELECTED_DOCS:
        file_path = os.path.join(INPUT_DIR, doc_name)

        #  check to prevent crashin if didnt typed or changed names of decisionss
        if not os.path.exists(file_path):
            print(f"Warnning: File not found: {file_path}.")
            continue

        # open clean JSON and get only  reasoning part
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        reasoning_text = data["segments"]["reasoning"]

        # 1. save just clean text
        txt_filename = doc_name.replace(".json", "_READABLE.txt")
        txt_path = os.path.join(TEXTS_DIR, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(f"--- DOCUMENT: {doc_name} ---\n\n")
            txt_file.write(reasoning_text)

        # 2. adding  empty row for this document into  CSV template where I will insert right chunks
        csv_data.append({
            "document_name": doc_name,
            "q1_context_quotes": "",
            "q2_penalty_quotes": "",
            "q3_moderation_quotes": ""
        })

    # --- 3. SAVE THE CSV TEMPLATE ---
    csv_path = os.path.join(EVAL_DIR, "golden_dataset_template.csv")

    # saving as CSV so I can easily open it in excell etc
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "document_name",
            "q1_context_quotes",
            "q2_penalty_quotes",
            "q3_moderation_quotes"
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    print("\n=== TEMPLATE CREATED SUCCESSFULLY ===")
    print(f"1. Go to '{TEXTS_DIR}' to read the clean texts.")
    print(f"2. Open '{csv_path}' in Excel and paste your short quotes.")
    print(
        "If you have multiple quotes for one question, separate them with a pipe character (|) like this: 'zmluva o dielo|žalovaný meškal'")


if __name__ == "__main__":
    main()