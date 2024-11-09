from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from  langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages.human import HumanMessage
from langchain_community.document_loaders.csv_loader import CSVLoader 
from langchain_community.document_loaders import TextLoader
api_key=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)
model = ChatGoogleGenerativeAI(model="gemini-pro")
embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

retriever_prompt=(
    "Given the following chat history and user question"
    "rephrase the question into a standalone format that captures all relevant context and details from the conversation."
    "Ensure the reformulated question is clear,"
    "specific, and can be understood independently without requiring reference to the chat history"
)

contextualize_prompt=ChatPromptTemplate.from_messages(
    [
        ('system',retriever_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

prompt = ChatPromptTemplate.from_template(
  """ 
 Answer the following question based only onthe provided context.
 Think step by step before providing the answer.
 <context>{context}</context>
 Question: {input}
"""
)

history=[]

def get_file_type(filename):
    _, ext = os.path.splitext(filename)
    if ext=='.pdf':
        loader = PyPDFLoader(path)
        return loader
    elif ext=='.csv':
        loader = CSVLoader(path, csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["name", "age", "sex","hobbies"],
    },)
        return loader
    elif ext=='.txt':
        loader = TextLoader(path)
        return loader

def load_to_vector(path):
    loader=get_file_type(path)
    documents = loader.lazy_load()
    text_document_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents_pdf=text_document_splitter.split_documents(documents)
    vectorstore=InMemoryVectorStore.from_documents(documents=documents_pdf,embedding=embedding)
    return vectorstore
def vector_question(vectorstore):
    retriever=vectorstore.as_retriever()
    history_aware=create_history_aware_retriever(model, retriever,contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    reg_chain=create_retrieval_chain(history_aware,question_answer_chain)
    return reg_chain

def input_to_response(question,reg_chain):
    message1=reg_chain.invoke({'input': question,'history': history})
    history.extend([HumanMessage(content=question),message1])
    return message1['answer']

def get_answer(path,input):
    vectorstore=load_to_vector(path)
    reg_chain=vector_question(vectorstore)
    return input_to_response(input,reg_chain)

st.title('AI-Powered Document Analysis and Retrieval')
uploaded_file=st.file_uploader("Chose a file",type=['pdf','txt','csv' ])
st.subheader('Enter Some Data')
user_input=st.text_input('Enter your Question')
button=st.button('Get Answer')
if button:
    if uploaded_file is not None:
        path=uploaded_file.name
        answer=get_answer(path,user_input)
        st.subheader('The Answer Is:')
        st.write(answer)
    else:
        st.subheader('No File Uploaded')
        st.write('Please upload a PDF file')

