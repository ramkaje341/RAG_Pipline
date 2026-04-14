# 🎬 Movie Recommendation System – Agentic RAG Pipeline

A **LangGraph-powered agentic RAG pipeline** for movie recommendations. The system retrieves candidates from a local FAISS vector store, verifies them using an LLM, and falls back to a live Tavily web search when local results don't match the query.

---

## 🏗️ Pipeline Architecture

```
retrieve_local → generate → extract_titles → verify ──pass──► END
                                                     ──fail──► internet_fallback → END
```

| Node | Role |
|---|---|
| `retrieve_local` | FAISS similarity search — top 3 movies from TMDB dataset |
| `generate` | Groq generates recommendations strictly from FAISS context |
| `extract_titles` | Regex parses movie titles from the LLM response |
| `verify` | Groq fact-checks each title against the query |
| `internet_fallback` | Tavily web search → Groq synthesizes from fetched results only |

The verify step is what makes this agentic — if FAISS returns irrelevant movies (e.g. querying by director when the index has no director metadata), the graph self-corrects and routes to a live web search instead of returning wrong results.

---

## 📈 Key Features

- **Agentic self-correction:** Verifies local results before returning them, falls back to internet search on failure
- **Live web search fallback:** Tavily search with Groq synthesis — grounded in fetched content, not LLM training data
- **Streaming output:** Responses stream word-by-word via `st.write_stream`
- **Agent transparency:** `st.status` panel shows which path the graph took and what was verified
- **Semantic retrieval:** FAISS + `all-MiniLM-L6-v2` embeddings over TMDB 5000 dataset

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM | Groq API (`llama-3.1-8b-instant`) |
| Vector store | FAISS (`faiss-cpu`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Web search | Tavily |
| UI | Streamlit |
| Data | TMDB 5000 Movies dataset |

---

## 📂 Repository Structure

```
├── app.py                # LangGraph pipeline + Streamlit UI
├── requirements.txt      # Project dependencies
├── .env                  # API keys (not committed)
├── tmdb_5000_movies.csv  # Raw movie dataset
└── movie_index/          # Auto-generated FAISS index
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ramkaje341/RAG_Pipeline.git
cd RAG_Pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- Groq API key: [console.groq.com](https://console.groq.com)
- Tavily API key: [app.tavily.com](https://app.tavily.com)

### 4. Run
```bash
streamlit run app.py
```

The FAISS index is built automatically on first run and saved to `movie_index/`.

---

## 🧪 Example Queries

| Query | Path taken |
|---|---|
| *Sci-fi movies about space* | Local FAISS (passes verify) |
| *Christopher Nolan movies* | Internet fallback (FAISS has no director metadata) |
| *Best Rajamouli films* | Internet fallback |
| *Romantic movies with sad ending* | Local FAISS |

---

## 👤 Author

**Sriram K**

---

## 📝 License

This project is intended for educational and analytical purposes.
