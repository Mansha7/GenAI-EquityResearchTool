import os
import streamlit as st
import pickle
import time
import requests
from langchain import HuggingFaceHub
from langchain_community.llms import huggingface_hub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import unstructured
from langchain.schema import Document

# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# session = requests.Session()
# retries = Retry(total=50, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
# session.mount("https://", HTTPAdapter(max_retries=retries))


from dotenv import load_dotenv
load_dotenv() #loads environment variables from env file

st.title("News Research Tool")
st.sidebar.title("News Article URLSs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder= st.empty()
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Replace with your desired model
    model_kwargs={"temperature": 0.5, "max_length": 100}
)

def fetch_url_content(urls):
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad responses
            content = response.text
            documents.append(Document(page_content=content, metadata={"source": url}))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
    return documents

#Getting vector database ready
if process_url_clicked:
    #load data
    #loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    #data = loader.load()
    data = fetch_url_content(urls)
    #st.write(data)

    #split data into chunks
    text_splitter= RecursiveCharacterTextSplitter(
        separators=[' ','\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Data...Started...")
    docs= text_splitter.split_documents(data)
    #doc_len= len(docs)
    #st.write("Number of documents:", doc_len) #fine till here

    #create embeddings and save to FAISS index
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"timeout": 120}  # Increase timeout to 120 seconds
        #huggingfacehub_api_token
    )
    main_placeholder.text("Embedding Vector Building...Started...")
    vectorstore_hf = FAISS.from_documents(docs, embeddings)
    
    time.sleep(2)

    #save faiss index to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

query = main_placeholder.text_input("Type your Question:")
if query:
    st.write("Processing...")
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            #{"answer:" & "source:"}
            res=chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(res["answer"])

            #Display sources if available
            sources = res.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for src in sources_list:
                    st.write(src)
    else:
        st.write("Could not fetch")