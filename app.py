import streamlit as st
import pandas as pd
import ast
import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -----------------------------
# CONFIG
# -----------------------------
INDEX_PATH = "movie_index"

st.set_page_config(page_title="Movie RAG App", layout="centered")
st.title("🎬 Movie Recommendation System (RAG)")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres', 'keywords']].dropna()

    def clean_json(col):
        return col.apply(lambda x: " ".join([i['name'] for i in ast.literal_eval(x)]))

    df['genres'] = clean_json(df['genres'])
    df['keywords'] = clean_json(df['keywords'])

    return df

df = load_data()


@st.cache_resource
def get_vectorstore(df):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )

    
    docs = []
    for _, row in df.iterrows():
        text = f"""
        Title: {row['title']}
        Genres: {row['genres']}
        Keywords: {row['keywords']}
        """
        docs.append(Document(page_content=text))

    vectorstore = FAISS.from_documents(docs, embedding)

    # ✅ Save index (important)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore

vectorstore = get_vectorstore(df)

# -----------------------------
# RETRIEVER (FASTER)
# -----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM (FASTER MODEL)
# -----------------------------
llm = Ollama(model="phi")  # ⚡ faster than mistral

# -----------------------------
# PROMPT (SHORT + OPTIMIZED)
# -----------------------------
prompt_template = """
You are a movie recommendation assistant.

Based on the following movies:
{context}

Answer the question:
{question}

Give:
- 3 to 5 movie recommendations
- One short reason for each
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -----------------------------
# RAG CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# -----------------------------
# UI
# -----------------------------
query = st.text_input("Ask for movie recommendations:")

if st.button("Recommend") and query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)

    st.subheader("🎯 Recommendations")
    st.write(result)