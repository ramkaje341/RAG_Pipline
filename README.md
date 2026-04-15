# 🎬 Movie Recommendation System – Agentic RAG Pipeline

A **LangGraph-powered agentic RAG pipeline** for movie recommendations. The system retrieves candidates from a local FAISS vector store, verifies them using an LLM, and falls back to a live Tavily web search when local results don't match the query. 

It now features an **Enhanced UI** that displays beautiful movie posters, ratings, and detailed overviews to enrich the recommendation experience!

---

## 🏗️ Pipeline Architecture

```text
retrieve_local → generate → extract_titles → verify ──pass──► END
                                                     ──fail──► internet_fallback → END
```

| Node | Role |
| :--- | :--- |
| **retrieve_local** | FAISS similarity search — fetches top 3 movies from the TMDB dataset. |
| **generate** | Groq generates recommendations strictly from the FAISS context. |
| **extract_titles** | Regex parses movie titles from the LLM response. |
| **verify** | Groq fact-checks each title against the user's original query. |
| **internet_fallback**| Tavily web search → Groq synthesizes recommendations from fetched live results. |

*The `verify` step is what makes this pipeline **agentic** — if FAISS returns irrelevant movies (e.g., querying by director when the index has no director metadata), the graph self-corrects and routes to a live web search instead of returning hallucinated or wrong results.*

---

## 📈 Key Features

* **🖼️ Enhanced UI:** Displays vibrant movie posters, user ratings, and detailed overviews fetched dynamically using the TMDB API.
* **🧠 Agentic Self-Correction:** Verifies local results before returning them; falls back to internet search on failure.
* **🌐 Live Web Search Fallback:** Uses Tavily search with Groq synthesis — grounded in fetched web content, not just LLM training data.
* **⚡ Streaming Output:** Responses stream word-by-word via `st.write_stream` for a fast, ChatGPT-like feel.
* **🔍 Agent Transparency:** An expandable `st.status` panel shows exactly which path the graph took and what was verified.
* **📂 Semantic Retrieval:** FAISS + `all-MiniLM-L6-v2` embeddings over the TMDB 5000 dataset.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | LangGraph (`StateGraph`) |
| **LLM** | Groq API (`llama-3.1-8b-instant`) |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Web Search** | Tavily API |
| **Metadata & Posters** | TMDB API |
| **UI** | Streamlit |
| **Data** | TMDB 5000 Movies dataset |

---

## 📂 Repository Structure

```text
├── app.py                # LangGraph pipeline + Streamlit UI
├── requirements.txt      # Project dependencies
├── .env                  # API keys (not committed)
├── tmdb_5000_movies.csv  # Raw movie dataset
└── movie_index/          # Auto-generated FAISS index directory
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
Create a `.env` file in the root of your project directory and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
TMDB_API_KEY=your_tmdb_api_key_here
```

**Where to get your keys:**
* **Groq API key:** [console.groq.com](https://console.groq.com/)
* **Tavily API key:** [app.tavily.com](https://app.tavily.com/)
* **TMDB API key:** [developer.themoviedb.org](https://developer.themoviedb.org/docs/getting-started)

### 4. Run the Application
```bash
streamlit run app.py
```
*Note: The FAISS vector index is built automatically on the first run and saved to the `movie_index/` directory.*

---

## 🧪 Example Queries

| Query | Path taken |
| :--- | :--- |
| *Sci-fi movies about space* | **Local FAISS** (passes verify step) |
| *Christopher Nolan movies* | **Internet fallback** (FAISS lacks director metadata) |
| *Best Rajamouli films* | **Internet fallback** (Self-corrected via web search) |
| *Romantic movies with sad ending* | **Local FAISS** (Accurately retrieved from dataset) |

---

## 👤 Authors

* **Sriram K**
* **Suhaas D**

---

## 📝 License

This project is intended for educational and analytical purposes.
