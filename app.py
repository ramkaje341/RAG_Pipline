import streamlit as st
import pandas as pd
import numpy as np
import ast
import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi


INDEX_PATH = "movie_index"

st.set_page_config(page_title="Movie RAG App", layout="centered")
st.title("🎬 Movie Recommendation System (Hybrid RAG)")


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


documents = []
doc_map = {}  # map text → index

for idx, row in df.iterrows():
    text = f"""
Title: {row['title']}
Overview: {row['overview']}
Genres: {row['genres']}
Keywords: {row['keywords']}
"""
    documents.append(text.strip())
    doc_map[text.strip()] = idx


tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


@st.cache_resource
def get_vectorstore(documents):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )

    docs = [Document(page_content=doc) for doc in documents]

    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore

vectorstore = get_vectorstore(documents)


def hybrid_retrieve(query, k=5):

    # ---- FAISS ----
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k*2)

    semantic_scores = {}
    for doc, score in docs_and_scores:
        text = doc.page_content.strip()
        if text in doc_map:
            idx = doc_map[text]
            semantic_scores[idx] = 1 / (1 + score)

    # ---- BM25 ----
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # ---- Combine ----
    final_scores = {}
    for i in range(len(documents)):
        final_scores[i] = (0.6 * semantic_scores.get(i, 0)) + (0.4 * bm25_scores[i])

    top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:k]

    return [documents[i] for i in top_indices]


llm = Ollama(model="phi3:mini")  


query = st.text_input("Ask for movie recommendations:")

if st.button("Recommend") and query:
    with st.spinner("Thinking..."):

        retrieved_docs = hybrid_retrieve(query)
        context = "\n\n".join(retrieved_docs)

        
        prompt = f"""
You are a movie recommendation system.

STRICT RULES:
- Do NOT generate code
- Do NOT explain logic
- ONLY recommend movies from the given list

Movies:
{context}

User Query:
{query}

Output format:

1. Movie Name - Reason
2. Movie Name - Reason
3. Movie Name - Reason

Only give the answer.
"""

        result = llm.invoke(prompt).strip()

  
    st.subheader("🎯 Recommendations")

    for line in result.split("\n"):
        if line.strip():
            st.markdown(f"- {line}")