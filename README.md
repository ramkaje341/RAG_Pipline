# 🎬 Movie Recommendation System – RAG Pipeline

A sophisticated **Retrieval-Augmented Generation (RAG)** solution designed to provide personalized movie recommendations and detailed explanations. By combining **FAISS** for high-speed similarity search and local LLMs via **Ollama**, this app delivers fast, context-aware suggestions based on the TMDB 5000 Movies dataset.

---

## 📈 Key Features

* **Hybrid Context:** Uses both movie descriptions and metadata (genres, keywords).
* **Natural Language Queries:** Ask like: "Movies like Interstellar".
* **Explainable Recommendations:** Provides reasons for each suggested movie.
* **Local Inference:** Runs completely offline using Ollama (no API cost).
* **Fast Retrieval:** FAISS enables efficient similarity search.

---

## 📊 Project Overview

This project addresses the "analysis paralysis" of choosing a movie by:
* **Semantic Search:** Understanding user intent (e.g., "movies about space travel and isolation") rather than just matching keywords.
* **Contextual Explanations:** Using an LLM to explain *why* a specific movie was recommended based on its plot and metadata.
* **Local Privacy:** Running the entire inference pipeline locally using Ollama and FAISS.

---

## 🏗️ System Architecture

The pipeline follows a streamlined RAG workflow inside a single application:

1.  **Data Preprocessing:** Cleaning the TMDB 5000 dataset using **Pandas** to extract titles, overviews, genres, and keywords.
2.  **Embedding Generation:** Converting movie metadata into high-dimensional vectors using **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`).
3.  **Vector Indexing:** Storing embeddings in **FAISS** for optimized, low-latency similarity retrieval.
4.  **Augmented Generation:** Injecting retrieved movie data into a structured prompt for **Ollama (Phi)** to generate a conversational response.

---

## 🛠️ Tech Stack

### ⚙️ Backend / AI Layer
* **Python:** Core programming language.
* **LangChain:** Framework for prompt chaining, retrieval management, and LLM integration.
* **FAISS:** High-performance vector database for similarity search.
* **SentenceTransformers:** Generates state-of-the-art text embeddings.

### 🤖 LLM Layer
* **Ollama:** Facilitates running large language models locally.
* **Phi:** Local, lightweight model used for generating fast natural language recommendations and summaries.

### 🌐 Frontend / UI
* **Streamlit:** Interactive web interface for user queries and displaying results.

---

## 📂 Repository Structure

```text
├── app.py                # Streamlit UI, Data Preprocessing, and FAISS indexing logic
├── requirements.txt      # Project dependencies
├── tmdb_5000_movies.csv  # Raw movie dataset
└── movie_index/          # Auto-generated local directory for the FAISS vector index
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.9+**
* **Ollama** installed and running locally on your machine.
* Pull the required Phi model via Ollama:  
  ```bash
  ollama pull phi
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/ramkaje341/RAG_Pipeline.git
cd RAG_Pipeline
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Usage
*Note: The vector index is automatically generated and saved when you first run the app, so you don't need a separate `vector_store.py` script!*

Launch the Streamlit app:
```bash
streamlit run app.py
```

---

## 🧪 Example Queries

Try asking the recommendation engine:
* *Movies like Interstellar*
* *Romantic movies with sad ending*
* *Action movies with war theme*
* *Feel good friendship movies*

---

## 👤 Author

**Sriram K**

---

## 📝 License

This project is intended for educational and analytical purposes.
