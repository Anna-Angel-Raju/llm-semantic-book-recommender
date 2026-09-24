# 📚 Semantic Book Recommender

An NLP-based book recommendation system that recommends books from natural-language descriptions using **semantic similarity, Hugging Face embeddings, ChromaDB, emotion analysis, and metadata filtering**.

##  Features

-  Natural-language semantic book search
-  Sentence embeddings using Hugging Face
- ️ Vector similarity search with ChromaDB
-  Emotion-based recommendation ranking
-  Category-based filtering
- ️ Book cover previews
-  Interactive Gradio web interface

##  How It Works

```text
Book Dataset
     ↓
Data Cleaning
     ↓
Emotion Analysis
     ↓
Book Descriptions
     ↓
Hugging Face Embeddings
     ↓
ChromaDB Vector Store
     ↓
User Query
     ↓
Semantic Similarity Search
     ↓
Category / Emotion Filtering
     ↓
Recommended Books
     ↓
Gradio Interface
```

##  Recommendation Pipeline

The user enters a natural-language query such as:

```text
A story about a troubled family across generations
```

The query is converted into an embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

ChromaDB then retrieves semantically similar book descriptions.

The retrieved books are mapped back to the main dataset using their **ISBN**, after which category and emotional preferences can be applied.

## Emotion Analysis

Book descriptions are analyzed using:

```text
j-hartmann/emotion-english-distilroberta-base
```

The model predicts:

- Anger
- Disgust
- Fear
- Joy
- Sadness
- Surprise
- Neutral

These scores are stored with each book and can be used to rank recommendations according to the selected emotional tone.

##  Tech Stack

- Python
- Pandas
- NumPy
- Hugging Face Transformers
- Sentence Transformers
- LangChain
- ChromaDB
- Gradio
- Jupyter Notebook

##  Project Structure

```text
book_recommender/
├── data-exploration.ipynb
├── text-classification.ipynb
├── sentiment-analysis.ipynb
├── vector-search.ipynb
│
├── gradio-dashboard.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/book_recommender.git
cd book_recommender
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
python gradio-dashboard.py
```

Gradio will provide a local URL such as:

```text
http://127.0.0.1:7860
```

Open it in your browser to use the recommender.

## 💡 Example Queries

```text
A story about forgiveness
```

```text
A troubled family dealing with secrets across generations
```

```text
A psychological story about loneliness
```

```text
A book about World War One
```

##  Key Design

The project combines **unstructured semantic information** with **structured metadata**:

```text
Semantic Search
      +
Category Filtering
      +
Emotion-based Ranking
      ↓
Book Recommendations
```

ISBN acts as the bridge between the vector database and the main Pandas dataset.

##  Limitations

- Semantic similarity does not guarantee factual matching.
- Queries about specific entities may retrieve broadly related books.
- Emotion scores depend on the classification model.
- Recommendation quality depends on the underlying dataset.

##  Future Improvements

- Hybrid keyword + semantic search
- Better entity-aware filtering
- Persistent ChromaDB storage
- Improved emotion aggregation
- Recommendation evaluation using Precision@K and NDCG@K
- Re-ranking using multiple recommendation signals

##  Learning Outcomes

This project provided hands-on experience with:

**NLP → Text Classification → Embeddings → Vector Databases → Semantic Search → Recommendation Systems → Gradio**

---

