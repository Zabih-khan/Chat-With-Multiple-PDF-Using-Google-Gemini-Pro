import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat Documents", page_icon=":notebook_with_decorative_cover:")   
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with PDF using Gemini💁")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat on PDFs :notebook:")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Generating response..."):
            response = user_input(user_question)

            # Add user input and bot response to chat history
            st.session_state.chat_history.append((user_question, response))
    
    for  question, response in st.session_state.chat_history:
        st.markdown(user_template.replace("{{MSG}}", f"{question}"), unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", f"{response}"), unsafe_allow_html=True)

    with st.sidebar:
        st.image("./assets/subheader.png")
        pdf_docs = st.file_uploader(
                "Upload your documents here and click on Process", accept_multiple_files=True)

        if st.button("Process :mag_right:"):
            with st.spinner("Processing"): 
                
                # get the pdf text:
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks:
                text_chunks = get_text_chunks(raw_text)

                # create vector store:
                vectorstore = get_vector_store(text_chunks)     
                
                # create conversation chain:
                st.session_state.conversation = get_conversational_chain()
                st.markdown("<h2 style='text-align: center; color: green;'>✔️</h2>", unsafe_allow_html=True) 




if __name__ == "__main__":
    main()
