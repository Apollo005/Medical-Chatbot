from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

PINECONE_API_KEY = "ab697d60-75e3-4cfe-88d9-83f88f8a025b"

def load_pdf(data):
    loader = DirectoryLoader(data, 
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")