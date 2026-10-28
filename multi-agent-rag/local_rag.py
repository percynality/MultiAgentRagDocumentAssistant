
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS



EMBEDDING_MODEL = "nomic-embed-text"


def build_vector_store(pdf_path: str):
    """Load a PDF, split it into chunks, and index them locally."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    print("1. Reading PDF...")
    pages = PyPDFLoader(str(path)).load()

    print("2. Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text was extracted. Check whether this PDF contains selectable text."
        )

    print(f"   Created {len(chunks)} chunks.")

    print("3. Creating local embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("4. Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("   Index created successfully!")
    return vector_store


def search_document(vector_store, question: str, k: int = 4):
    """Retrieve relevant PDF passages and their page numbers."""

    results = vector_store.similarity_search(question, k=k)

    print("\n--- RETRIEVED PASSAGES ---")

    if not results:
        print("No relevant passages found.")
        return results

    for index, doc in enumerate(results, start=1):
        # PyPDFLoader uses zero-based page numbers.
        page_number = doc.metadata.get("page", 0) + 1

        print(f"\nPassage {index} | PDF page {page_number}")
        print(doc.page_content[:700])

    return results


if __name__ == "__main__":
    pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

    try:
        vector_store = build_vector_store(pdf_path)

        while True:
            question = input(
                "\nAsk a question about your PDF, or type 'exit': "
            ).strip()

            if question.lower() == "exit":
                break

            if question:
                search_document(vector_store, question)

    except Exception as exc:
        print(f"\nError: {exc}")
        print("Check that Ollama is running and both models are installed.")
        



