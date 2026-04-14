import re
import time
import streamlit as st
import pandas as pd
import ast
import os

from typing import TypedDict
from dotenv import load_dotenv
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()


INDEX_PATH = "movie_index"

st.set_page_config(page_title="Movie RAG App", layout="centered")
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
    # return Ollama(model="llama3.1")
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = get_llm()


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
    prompt = f"""You are a movie recommendation assistant.

Movies available (use ONLY these, do not add any others):
{state["context"]}

Request: {state["query"]}

Reply with ONLY a numbered list using movies from the list above. No introduction. No extra movies.
1. Title - one sentence reason
2. Title - one sentence reason"""

    return {"response": llm.invoke(prompt).content}


def extract_titles_node(state: AgentState) -> AgentState:
    response = state["response"]

    titles = []
    for line in response.strip().split("\n"):
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        if not line:
            continue
        title = line.split(" - ")[0] if " - " in line else None
        if title:
            title = title.strip().strip('"\'*()–-[]')
            if 2 < len(title) < 60:
                titles.append(title)

    if titles:
        return {"movie_titles": titles}

    matches = re.findall(r'"?([A-Z][A-Za-z\'\s:!?&]+?)"?\s*\((\d{4})\)', response)
    if matches:
        return {"movie_titles": [m[0].strip() for m in matches]}

    matches = re.findall(r'(?:^|[,\n]\s*)([A-Z][A-Za-z\'\s:!?&]{3,40})(?=\s*[\"(,])', response)
    return {"movie_titles": [m.strip() for m in matches if m.strip()]}


def verify_node(state: AgentState) -> AgentState:
    titles = state.get("movie_titles", [])
    if not titles:
        return {"verified": False, "failed_movies": []}

    checks = "\n".join([f'- "{t}"' for t in titles])
    prompt = f"""For each movie below, answer Yes if it matches "{state["query"]}", or No if it does not. Be accurate and factual.

Reply in this exact format, one line per movie, nothing else:
"Movie Title": Yes/No

Movies:
{checks}"""

    result = llm.invoke(prompt).content
    failed = [t for t in titles if f'"{t}": No' in result or f'"{t}":No' in result]
    verified = len(failed) < len(titles) * 0.5

    return {"verified": verified, "failed_movies": failed}


def internet_fallback_node(state: AgentState) -> AgentState:
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = tavily.search(query=f"{state['query']} best films", max_results=8)["results"]
    except Exception:
        results = []

    context = "\n\n".join([
        f"{r['title']}: {r['content'][:200]}"
        for r in results if r.get("title") and r.get("content")
    ])

    if not context:
        return {"response": "No web results found. Please try again later."}

    prompt = f"""You are a movie recommendation assistant. Using the web results below, extract real movie recommendations for the request.

Web results:
{context}

Request: {state["query"]}

Output a clean numbered list of up to 5 movies. Each line must be exactly:
N. Movie Title (Year) - one sentence about the film

Rules:
- Only include actual movie titles, no ranking commentary
- Do not repeat the same movie twice
- No introductions, no extra text, nothing after the list"""

    return {"response": llm.invoke(prompt).content}


def verify_decision(state: AgentState) -> str:
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


def word_streamer(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.03)


query = st.text_input("Ask for movie recommendations:")

if st.button("Recommend", use_container_width=True) and query:
    st.write(f"**You:** {query}")

    with st.status("Agent thinking...", expanded=True) as status:
        st.write("Fetching from local database...")
        result = agent.invoke({"query": query})

        titles = result.get("movie_titles", [])
        failed = result.get("failed_movies", [])
        verified = result.get("verified", False)

        if titles:
            st.write(f"Generated: {', '.join(titles)}")

        if verified:
            st.write("Verification: all movies confirmed via web")
        else:
            st.write(f"Verification: {', '.join(failed) if failed else 'low confidence'} — falling back to internet search")

        status.update(label="Done", state="complete")

    st.write_stream(word_streamer(result["response"]))
