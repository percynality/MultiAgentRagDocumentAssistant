
from pathlib import Path
from pypdf import PdfReader


def get_document_metadata(pdf_path: str) -> dict:
    """Return basic metadata about a PDF."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    metadata = reader.metadata

    return {
        "filename": path.name,
        "title": metadata.title if metadata and metadata.title else "Unknown",
        "author": metadata.author if metadata and metadata.author else "Unknown",
        "page_count": len(reader.pages),
    }


def metadata_agent(pdf_path: str, question: str) -> str:
    """Answer basic document metadata questions."""

    info = get_document_metadata(pdf_path)
    question_lower = question.lower()

    if "author" in question_lower or "wrote" in question_lower:
        return f"Document author: {info['author']}"

    if "title" in question_lower or "name" in question_lower:
        return f"Document title: {info['title']}"

    if "page" in question_lower or "long" in question_lower:
        return f"Total pages: {info['page_count']}"

    return (
        f"Filename: {info['filename']}\n"
        f"Title: {info['title']}\n"
        f"Author: {info['author']}\n"
        f"Pages: {info['page_count']}"
    )


if __name__ == "__main__":
    pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')
    question = input("Ask about the document's metadata: ").strip()

    try:
        print("\n--- METADATA AGENT ---")
        print(metadata_agent(pdf_path, question))
    except Exception as exc:
        print(f"Error: {exc}")
