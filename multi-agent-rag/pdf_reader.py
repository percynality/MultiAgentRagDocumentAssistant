from pathlib import Path
from pypdf import PdfReader


def read_pdf(pdf_path: str) -> None:
    path = Path(pdf_path)

    if not path.is_file():
        print(f"File not found: {path}")
        return

    try:
        reader = PdfReader(str(path))
        metadata = reader.metadata

        print("\n--- PDF INFORMATION ---")
        print(f"Filename: {path.name}")
        print(f"Title: {metadata.title if metadata else 'Unknown'}")
        print(f"Author: {metadata.author if metadata else 'Unknown'}")
        print(f"Total pages: {len(reader.pages)}")

        print("\n--- FIRST PAGE TEXT ---")
        if reader.pages:
            text = reader.pages[0].extract_text() or ""
            print(text[:2000] if text.strip()
                  else "No extractable text found.")

    except Exception as exc:
        print(f"Could not read the PDF: {exc}")


if __name__ == "__main__":
    pdf_path = input("Enter the full path to your PDF: ")
    read_pdf(pdf_path.strip().strip('"'))
    