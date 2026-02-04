import os
import re
import io
from pypdf import PdfReader

# --- CONFIGURATION ---
# Enter the folder name where you have downloaded the 1500 PDFs
INPUT_DIR = "dataset_zmluvna_pokuta_filtered_2"

# Keywords for "Holy Grail" (Reasonableness / Moderation right)
# We are looking for Section 300 of the Commercial Code or talk about unreasonableness
REGEX_MODERACIA = (
    r"§\s*300|"  # Section about penalty reduction
    r"moderačn[ééhoau]|"  # Moderation right/authority
    r"neprimerane\s+vysok|"  # Unreasonably high
    r"zníž[ilne]\s+pokut"  # Court reduced fine / reduction of fine
)

# Keywords for "Junk" (what we don't want)
REGEX_BALAST = (
    r"poriadkov[áúu]\s+pokut|"  # Procedural fine
    r"blokov[áúu]\s+pokut|"  # On-the-spot fine
    r"priestup"  # Misdemeanor / Offense
)


def analyze_folder():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Folder '{INPUT_DIR}' does not exist.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files. Starting analysis...\n")

    count_moderacia = 0
    count_validita = 0
    count_balast = 0

    # Lists to save filenames for output
    list_moderacia = []

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)

        try:
            text = ""
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                # We read all pages (or specific ones if you want to optimize)
                # Loop through all pages
                for page in reader.pages:
                    text += page.extract_text() + " "

            # 1. Is it junk?
            if re.search(REGEX_BALAST, text, re.IGNORECASE):
                count_balast += 1
                # print(f"🗑️  Junk: {filename}")
                continue

            # 2. Is it about amount/reasonableness (Section 300)?
            if re.search(REGEX_MODERACIA, text, re.IGNORECASE):
                count_moderacia += 1
                list_moderacia.append(filename)
                print(f"MODERATION (Section 300): {filename}")
                continue

            # 3. If neither, it is likely just a common dispute about payment/validity
            count_validita += 1

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("\n" + "=" * 40)
    print(f"ANALYSIS RESULT OF {len(files)} DECISIONS")
    print("=" * 40)
    print(f"Decisions on REASONABLENESS (Section 300, reduction): {count_moderacia}")
    print(f"Common disputes (claim, validity, payment):       {count_validita}")
    print(f"Probable junk (procedural fines):                   {count_balast}")
    print("=" * 40)

    if count_moderacia > 0:
        print("\nI recommend starting by reading these files (saved in 'zoznam_moderacia.txt'):")
        with open("zoznam_moderacia.txt", "w") as f:
            for item in list_moderacia:
                f.write(f"{item}\n")
        print("(List saved to file)")


if __name__ == "__main__":
    analyze_folder()