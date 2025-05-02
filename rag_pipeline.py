from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader # Loads text documents.
from langchain_community.vectorstores import Chroma # A vector store for efficiently storing and searching vector embeddings.
from langchain_community.embeddings import SentenceTransformerEmbeddings # Converts text chunks into vector embeddings using a pre-trained model.
from langchain.text_splitter import RecursiveCharacterTextSplitter # Splits large text into smaller, manageable chunks.
import os # Used for operating system interactions (though not used explicitly in this script).
import shutil


embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load & split
def load_pdf(pdf):
    loader = PyPDFLoader(pdf)  # :contentReference[oaicite:17]{index=17}
    docs = loader.load()
    return docs


def retrieve_embedding(pdf):
    docs = load_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits
    
def save_embedding():
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)

    texts = retrieve_embedding(pdf)

    vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding_model,
            # persist_directory=vector_db_path,
        )

    # vectordb.persist()
    return vectordb.as_retriever()

if __name__ == "__main__":
    save_embedding()