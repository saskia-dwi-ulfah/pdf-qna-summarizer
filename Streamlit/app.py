import shutil
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from utils.function import format_docs, display_how_to
from utils.template import template


def process_file(uploaded_file, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        shutil.copyfileobj(uploaded_file, tmpfile)
        tmpfile_path = tmpfile.name

    loader = PyPDFLoader(tmpfile_path, extract_images=False)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))
    return vectorstore

def clear_all():
    keys = list(st.session_state.keys())

    for key in keys:
        del st.session_state[key]



with st.sidebar:
    st.sidebar.title('PDF Q&A and Summarizer')
    st.button("How to Use The App", on_click = display_how_to)
    st.sidebar.markdown('---')
    api_key = st.text_input("Input your Open AI API key")

    print(st.session_state)

    with st.spinner('Uploading your file...'):
        uploaded_file = st.file_uploader("Upload your PDF file", type = ["pdf"], on_change = clear_all)

    if uploaded_file is None: 
        st.warning("Please upload your PDF file.")
    else:
        if 'vectorstore' not in st.session_state:
            st.session_state['vectorstore'] = process_file(uploaded_file, api_key)

        st.session_state['option'] = st.sidebar.selectbox(
                                        'Choose Menu',
                                        ('QnA', 'Summarizer')
                                    )
    
if 'vectorstore' in st.session_state:
    # vectorstore = st.session_state.vectorstore
    # option = st.session_state.option

    # display content based on the selected option
    if st.session_state.option == 'QnA':
        st.title("Ask The PDF 📑🔮🤔")
        st.caption("Powered by Open AI GPT 3.5 Turbo")

        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatOpenAI(model_name = 'gpt-3.5-turbo-0125', temperature = 0, openai_api_key = api_key)
        custom_rag_prompt = PromptTemplate.from_template(template)
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

    elif st.session_state.option == 'Summarizer':
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatOpenAI(model_name = 'gpt-3.5-turbo-0125', temperature = 0, openai_api_key = api_key)

        st.title("Summarize The PDF 📑✍️")
        st.caption("Powered by Open AI GPT 3.5 Turbo")

        qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                    chain_type = "stuff",
                                    retriever = retriever,
                                    return_source_documents = True,
                                    verbose = False)

        chain_result = qa_chain("Give me the summary in general!")
        answer = chain_result["result"]

        st.write(answer)

        # @st.cache_data
        # def cache_summarizer():
        #     qa_chain = RetrievalQA.from_chain_type(llm = llm,
        #                             chain_type = "stuff",
        #                             retriever = retriever,
        #                             return_source_documents = True,
        #                             verbose = False)

        #     chain_result = qa_chain("Give me the summary in general!")
        #     answer = chain_result["result"]

        #     st.write(answer)

        # cache_summarizer()
    



    
