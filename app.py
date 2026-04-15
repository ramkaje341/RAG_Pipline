import re
import streamlit as st
import pandas as pd
import ast
import os
import requests

from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

INDEX_PATH = "movie_index"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(page_title="Movie RAG App", layout="wide")
st.title("🎬 Movie Recommendation System (RAG)")


@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres', 'keywords']].dropna()

    def clean_json(col):
        return col.apply(lambda x: " ".join([i['name'] for i in ast.literal_eval(x)]))

    df['genres'] = clean_json(df['genres'])
    df['keywords'] = clean_json(df['keywords'])

    return df



@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)

    df = load_data()
    docs = []
    for _, row in df.iterrows():
        text = f"Title: {row['title']}\nGenres: {row['genres']}\nKeywords: {row['keywords']}"
        docs.append(Document(page_content=text))

    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore



@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = get_llm()


def clean_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    title = title.split(" - ")[0]
    return title.strip()



def get_movie_details(title):
    if not TMDB_API_KEY:
        return None

    clean = clean_title(title)

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": clean}

    try:
        data = requests.get(url, params=params).json()

        if data.get("results"):
            for movie in data["results"]:
                if clean.lower() in movie.get("title", "").lower():
                    return {
                        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                        "rating": movie.get("vote_average"),
                        "overview": movie.get("overview")
                    }

            movie = data["results"][0]
            return {
                "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                "rating": movie.get("vote_average"),
                "overview": movie.get("overview")
            }
    except:
        return None

    return None



class AgentState(TypedDict):
    query: str
    context: str
    response: str
    movie_titles: list
    verified: bool
    failed_movies: list



def retrieve_local_node(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}


def generate_node(state: AgentState) -> AgentState:
    prompt = f"""
You are a movie recommendation assistant.

Movies available:
{state["context"]}

Request: {state["query"]}

Return ONLY:
1. Movie Name - Reason
2. Movie Name - Reason
"""
    return {"response": llm.invoke(prompt).content}


def extract_titles_node(state: AgentState) -> AgentState:
    titles = []

    for line in state["response"].split("\n"):
        match = re.match(r"\d+\.\s*(.*?)\s*-", line)
        if match:
            titles.append(clean_title(match.group(1)))

    return {"movie_titles": titles}


def verify_node(state: AgentState) -> AgentState:
    titles = state.get("movie_titles", [])
    if not titles:
        return {"verified": False, "failed_movies": []}

    checks = "\n".join([f'- "{t}"' for t in titles])
    prompt = f"""
For each movie below, answer Yes or No if it matches "{state["query"]}".

Movies:
{checks}

Format:
"Movie": Yes/No
"""

    result = llm.invoke(prompt).content
    failed = [t for t in titles if f'"{t}": No' in result]

    return {"verified": len(failed) < len(titles) * 0.5, "failed_movies": failed}


def internet_fallback_node(state: AgentState) -> AgentState:
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = tavily.search(query=f"{state['query']} movies", max_results=8)["results"]
    except:
        results = []

    context = "\n\n".join([r['title'] for r in results if r.get("title")])

    prompt = f"""
Extract ONLY movie titles:

{context}

Return:
1. Movie Name (Year)
2. Movie Name (Year)
"""

    return {"response": llm.invoke(prompt).content}


def verify_decision(state: AgentState):
    return "pass" if state.get("verified") else "fail"



@st.cache_resource
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_local", retrieve_local_node)
    graph.add_node("generate", generate_node)
    graph.add_node("extract_titles", extract_titles_node)
    graph.add_node("verify", verify_node)
    graph.add_node("internet_fallback", internet_fallback_node)

    graph.set_entry_point("retrieve_local")

    graph.add_edge("retrieve_local", "generate")
    graph.add_edge("generate", "extract_titles")
    graph.add_edge("extract_titles", "verify")

    graph.add_conditional_edges("verify", verify_decision, {
        "pass": END,
        "fail": "internet_fallback",
    })

    graph.add_edge("internet_fallback", END)

    return graph.compile()


agent = build_graph()


query = st.text_input("Ask for movie recommendations:")

if st.button("Recommend") and query:

    with st.spinner("🤖 Thinking... Please wait"):
        result = agent.invoke({"query": query})
    titles = result.get("movie_titles", [])

    st.subheader("🎯 Recommendations")

    if titles:
        cols = st.columns(len(titles))

        for i, title in enumerate(titles):
            details = get_movie_details(title)

            with cols[i]:
                if details and details["poster"]:
                    st.image(details["poster"], width=200)

                st.markdown(f"### {title}")

                if details:
                    st.write(f"⭐ Rating: {details['rating']}")
                    if details["overview"]:
                        st.caption(details["overview"][:150] + "...")
                else:
                    st.write("Details not found")

    st.divider()

    st.subheader("💬 Explanation")

    for line in result["response"].split("\n"):
        if line.strip():
            st.markdown(f"- {line}")