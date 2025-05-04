from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, UnstructuredWordDocumentLoader # Loads text documents.
from langchain_community.vectorstores import Chroma, Weaviate # A vector store for efficiently storing and searching vector embeddings.
from langchain_community.embeddings import SentenceTransformerEmbeddings # Converts text chunks into vector embeddings using a pre-trained model.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # Splits large text into smaller, manageable chunks.
import os # Used for operating system interactions (though not used explicitly in this script).
import shutil

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure

from dotenv import load_dotenv()

load_dotenv()

YOUR_WEAVIATE_KEY = os.getenv("YOUR_WEAVIATE_KEY")
YOUR_WEAVIATE_CLUSTER = os.getenv("YOUR_WEAVIATE_CLUSTER")
YOUR_HUGGINGFACE_KEY = os.getenv("YOUR_HUGGINGFACE_KEY")


embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load & split
def load_pdf(pdf):
    loader = DirectoryLoader("./PDF", glob="**/*.pdf")  # :contentReference[oaicite:17]{index=17}
    docs = loader.load()
    print(f"You have {len(docs)} documents loaded.")
    print(f"You have {len(docs[0].page_content)} characters in your documents")
    return docs


def retrieve_embedding(pdf):
    docs = load_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits
    
def save_embedding_chroma():
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

def save_embedding_weaviate(pdf):
    headers = {
    "X-HuggingFace-Api-Key": YOUR_HUGGINGFACE_KEY,
    }

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                       # `weaviate_url`: your Weaviate URL
        auth_credentials=Auth.api_key(weaviate_key),      # `weaviate_key`: your Weaviate API key
        headers=headers
    )

    client.collections.create(
    "Book-Chatbot",
    vectorizer_config=[
        Configure.NamedVectors.text2vec_huggingface(
            name="title_vector",
            source_properties=["title"],
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    ],
    # Additional parameters not shown
    )

    collection = client.collections.get("Book-Chatbot")

    docs = load_pdf(pdf)
    text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
    texts, meta = list(zip(*text_meta_pair))

    with collection.batch.fixed_size(batch_size=200) as batch:
        batch.add_object(texts, meta)

    client.close()

    return collection

doc = save_embedding_weaviate(pdf)
doc = collection.similiarity_search(query, top_k=20)



if __name__ == "__main__":
    save_embedding_chroma()