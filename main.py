from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
##from langchain.chains import RetrievalQA
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


PINECONE_API_KEY = "ab697d60-75e3-4cfe-88d9-83f88f8a025b"
PINECONE_API_ENV = "serverless"


def load_pdf(data):
    loader = DirectoryLoader(data, 
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")

def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

embeddings

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name="med-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
        )
    )

index = pc.Index(index_name)

docs_chunks =[t.page_content for t in text_chunks]

vectorstore = PineconeVectorStore(
index=index,
pinecone_api_key = PINECONE_API_KEY,
embedding=embeddings,
namespace="medicalChatBot",
index_name='med-chatbot'
)

vectorstore.add_texts(texts=[t.page_content for t in text_chunks])


        

query = "What are Allergies"
result = index.query(queries=[query], top_k=3)
print("Result", result)


