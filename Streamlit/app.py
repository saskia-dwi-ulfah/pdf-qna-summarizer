import os
import shutil
import tempfile
import streamlit as st

from utils.function import format_docs, display_how_to
from utils.template import template

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import shutil
import tempfile
import streamlit as st

from utils.function import format_docs, display_how_to
from utils.template import template

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# Add a title to the sidebar
with st.sidebar:

    st.sidebar.title('Sidebar Selection')

    st.button("How to Use The App", on_click = display_how_to)

    st.sidebar.markdown('---')
    
    option = st.sidebar.selectbox(
        'Choose Menu',
        ('Home', 'QnA', 'Summarizer')
    )

    st.sidebar.markdown('---')

    api_key = st.text_input("Input your Open AI API key")

    st.sidebar.markdown('---')

    uploaded_file = st.file_uploader("Upload your PDF file", type = ["pdf"])

# Display content based on the selected option
if option == 'Home':
    st.title('Home Page')
    st.write('Welcome to the Home Page!')

elif option == 'QnA':
    st.title("Ask The PDF 📑🔮🤔")
    st.caption("Powered by Open AI GPT 3.5 Turbo")

    if uploaded_file is not None: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(uploaded_file, tmpfile)
            tmpfile_path = tmpfile.name

        loader = PyPDFLoader(tmpfile_path, extract_images = False) # error when load rapidocr-onnxruntime
        docs = loader.load()

        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Save
        vectorstore = Chroma.from_documents(documents = splits, 
                                    embedding = OpenAIEmbeddings(openai_api_key = api_key))

        # 4. Retrieve and generate 
        retriever = vectorstore.as_retriever()
        custom_rag_prompt = PromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name = 'gpt-3.5-turbo-0125', temperature = 0, openai_api_key = api_key)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        ) 

        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask the PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = rag_chain.invoke(prompt)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

elif option == 'Summarizer':
    st.title("Summarize The PDF 📑🔮🤔")
    st.caption("Powered by Open AI GPT 3.5 Turbo")

    if uploaded_file is not None: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            # Copy the uploaded file to the temporary file
            shutil.copyfileobj(uploaded_file, tmpfile)
            tmpfile_path = tmpfile.name

        loader = PyPDFLoader(tmpfile_path, extract_images = False) # error when load rapidocr-onnxruntime
        docs = loader.load()

        text_splitter = CharacterTextSplitter(separator = "\n", 
                                        chunk_size = 1000, 
                                        chunk_overlap = 10, 
                                        length_function = len)
        text = text_splitter.split_text(docs)

        embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

        vectordb = Chroma(persist_directory = "assets/chroma_db", embedding_function = embedding_function)

        load_dotenv()

        retriever = vectordb.as_retriever()
        llm = OpenAI(temperature = 0.9)

        qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                            chain_type = "stuff",
                                            retriever = retriever,
                                            return_source_documents = True,
                                            verbose = False)

        chain_result = qa_chain("Can you give me the summary")
        answer = chain_result["result"]
        print(answer)

# st.sidebar.markdown('---')

# st.sidebar:
#     api_key = st.text_input("Input your Open AI API key")
#     uploaded_file = st.file_uploader("Upload your PDF file", type = ["pdf"])