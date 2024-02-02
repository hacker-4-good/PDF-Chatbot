import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub 
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

# Passing uploaded pdf document to PyPDFLoader
def pdf_loader(pdf):
    if pdf:
        temp = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(pdf.read())
            loader = PyPDFLoader(file_path=temp_pdf.name)
            pdf_content = loader.load()
            temp = pdf_content
        
        os.remove(temp_pdf.name)
        return temp
    

# RAG chain for sending the loaded document into the llm for genetrating the response 
def RAG_chain(document, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(documents=document)
    vectorstore = Chroma.from_documents(
        documents = splits, 
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    )
    retriever = vectorstore.as_retriever() 
    prompt = hub.pull('rlm/rag-prompt')
    llm = GoogleGenerativeAI(
        model='models/text-bison-001'
    )
    rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain.invoke(query)


def PDFChatbot(pdf, query):
    document = pdf_loader(pdf)
    return RAG_chain(document = document, query = query)
    

# main function of the program comprises of basic UI for user interaction, where the end user ask the prompt query related to the pdf uploaded .....
if __name__=='__main__':
        
    try:
        st.title("PDF Chatbot 📚")
        with st.sidebar:
            st.title('PDF Chatbot 📃')
            google_api = st.text_input('Enter Google AI API-KEY:', type='password')
            if not (google_api.startswith('AI') and len(google_api)==39):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
            st.markdown("You can make your API Token key from here → [Link](https://makersuite.google.com/app/apikey)")
        os.environ['GOOGLE_API_KEY'] = google_api
        pdf = st.file_uploader('Upload PDF file', type='pdf')
        prompt = st.text_input('Enter the question you want to ask to the LLM', placeholder='Enter prompt your here.....')
        result = PDFChatbot(pdf = pdf, query = prompt)
        if st.button("Submit"):
            st.balloons()
            st.write(result)
    except:
        pass
