import os
import re
import io
from pypdf import PdfReader

# Second pass filter. After scraper downloads ~1500 PDFs, this script
# reads them and tries to find the ones really about § 300 / moderation
# (judge reducing the penalty).

# --- CONFIG ---
# Folder where I downloaded the PDFs in previous step
INPUT_DIR = "dataset_zmluvna_pokuta_filtered_2"

# Keywords for "holy grail" (reasonableness / moderation)
# Looking for § 300 ObchZ or talk about unreasonable penalty
REGEX_MODERACIA = (
    r"§\s*300|"  # paragraph about penalty reduction
    r"moderačn[ééhoau]|"  # moderation right
    r"neprimerane\s+vysok|"  # "unreasonably high"
    r"zníž[ilne]\s+pokut"  # court reduced penalty
)

# Keywords for "junk" - what I dont want
REGEX_BALAST = (
    r"poriadkov[áúu]\s+pokut|"  # poriadkova pokuta (procedural fine)
    r"blokov[áúu]\s+pokut|"  # blokova pokuta (on-the-spot fine)
    r"priestup"  # priestupok (misdemeanor)
)


def analyze_folder():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: folder '{INPUT_DIR}' does not exist")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files. Starting analysis...\n")

    count_moderacia = 0
    count_validita = 0
    count_balast = 0

    # save filenames for output
    list_moderacia = []

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)

        try:
            text = ""
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                # read all pages
                for page in reader.pages:
                    text += page.extract_text() + " "

            # 1. is it junk?
            if re.search(REGEX_BALAST, text, re.IGNORECASE):
                count_balast += 1
                # print(f"junk: {filename}")
                continue

            # 2. is it about amount/reasonableness (§ 300)?
            if re.search(REGEX_MODERACIA, text, re.IGNORECASE):
                count_moderacia += 1
                list_moderacia.append(filename)
                print(f"MODERATION (§ 300): {filename}")
                continue

            # 3. otherwise its probably common dispute about payment/validity
            count_validita += 1

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("\n" + "=" * 40)
    print(f"ANALYSIS RESULT OF {len(files)} DECISIONS")
    print("=" * 40)
    print(f"About REASONABLENESS (§ 300, reduction): {count_moderacia}")
    print(f"Common disputes (claim, validity, payment): {count_validita}")
    print(f"Junk (procedural fines):                 {count_balast}")
    print("=" * 40)

    if count_moderacia > 0:
        print("\nStart by reading these files (saved in 'zoznam_moderacia.txt'):")
        with open("zoznam_moderacia.txt", "w") as f:
            for item in list_moderacia:
                f.write(f"{item}\n")
        print("(list saved to file)")


if __name__ == "__main__":
    analyze_folder()
