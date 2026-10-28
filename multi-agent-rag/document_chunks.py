from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pdf_path: str):
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Load text from every page.
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    # Divide page text into smaller, overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(pages)

    print(f"\nPDF: {path.name}")
    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    # Display the first few chunks.
    for index, chunk in enumerate(chunks[:3], start=1):
        page_number = chunk.metadata.get("page", 0) + 1

        print(f"\n--- Chunk {index} | Page {page_number} ---")
        print(chunk.page_content[:500])

    return chunks


if __name__ == "__main__":
    pdf_path = input("Enter the full path to your PDF: ")