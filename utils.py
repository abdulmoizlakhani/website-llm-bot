from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import  RecursiveUrlLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_transformers import Html2TextTransformer
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import streamlit as st

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY') )
load_dotenv()  

def get_vectorstore_from_url(url, website_name):
    loader = RecursiveUrlLoader(url)
    pages = loader.load()
    tt=Html2TextTransformer()
    document=tt.transform_documents(pages)
 
    text_splitter =RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )

    document_chunks = text_splitter.split_documents(document)
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    index_name="chatbots"
    
    vector_store = PineconeVectorStore.from_documents(document_chunks , index_name=index_name , embedding=embeddings , namespace=website_name)

    return vector_store

def retrieve_answers(vector_store,query):

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))

    template = """You are a chatbot and allowed to answer the question either using the information provided below and your own knowledge, answer the question:
    Context from vector database: {context}
    Question: {question}
    """

    template_qa = """Answer the following question:
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    prompt_qa =  ChatPromptTemplate.from_template(template_qa)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    data = retriever.invoke(query)

    if len(data) == 0 :
        rag_chain = ( 
             {"question": RunnablePassthrough()} 
             | prompt_qa
        | llm
    )
    else:
        rag_chain = (
        { "context": retriever , "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response=rag_chain.invoke(query)

    return response
  

def get_context_retriever_chain(query , website_name):
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    
    Pinecone(api_key=os.environ.get('PINECONE_API_KEY') )
    index_name="chatbots"

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings , namespace=website_name)

    similar_docs = retrieve_answers(vector_store , query)

    return similar_docs

def namespace_exists(namespace):
    index = pc.Index("chatbots")
    namespaces = index.describe_index_stats()['namespaces']
    return namespace in namespaces

def all_name_spaces():
    index = pc.Index("chatbots")
    namespaces = index.describe_index_stats()['namespaces']
    if len(namespaces) == 0:
        return ""
    else:
        return namespaces.keys()