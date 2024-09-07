from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time

# Models
EMBEDDING = "mistral"
LOCAL_MODEL = "llama3"

# Store uploaded PDFs
if not os.path.exists('files'):
    os.mkdir('files')

# Store vector store data
if not os.path.exists('data'):
    os.mkdir('data')

# Initialize session state - store information across different user interactions during lifetime of a session

# Initialize it with a default template for the chatbot
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

# that uses 'history', 'context', and 'question' as input variables
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template)

# to store conversation history and return it during interactions
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

# 'embedding_function' uses OllamaEmbeddings to embed the data and persist it in the 'data' directory  
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='data',
                        embedding_function=OllamaEmbeddings(
                            model=EMBEDDING))

# using a local model with streaming output callbacks for real-time responses 
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                model=LOCAL_MODEL,
                verbose=True,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]))

# initialize it as an empty list to store user-chatbot conversation history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Set the title
st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

# Iterates over the chat history stored in chat_history and displays each messages in Streamlit
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()

            # Loads the PDF into memory
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model=EMBEDDING)
            )

            # Saves the vector store to data folder
            st.session_state.vectorstore.persist()

    # Converts the Chroma vector store into retriever object
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    # Initialize the QA chain - sets up process where the user query gets passed to a language model, 
    # which retrieves relevent documents using a retriever and generates an answer
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)


else:
    st.write("Please upload a PDF file.")