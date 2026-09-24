import pandas as pd
import numpy as np

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

import gradio as gr


# --------------------------------------------------
# 1. LOAD BOOK DATASET
# --------------------------------------------------

books = pd.read_csv("books_with_emotions.csv")

# Keep ISBN as string because it is an identifier, not a number
books["isbn13"] = (
    books["isbn13"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)


# --------------------------------------------------
# 2. CREATE LARGE THUMBNAIL URL
# --------------------------------------------------

books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)


# --------------------------------------------------
# 3. LOAD TAGGED DESCRIPTIONS
# --------------------------------------------------

raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

print("Descriptions loaded!")


# Each line represents one book
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)


# --------------------------------------------------
# 4. CREATE HUGGING FACE EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 5. CREATE CHROMA VECTOR DATABASE
# --------------------------------------------------

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)


# --------------------------------------------------
# 6. SEMANTIC RECOMMENDATION FUNCTION
# --------------------------------------------------

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Find books semantically similar to the user's query
    recs = db_books.similarity_search(
        query,
        k=initial_top_k
    )

    # Extract ISBN from each retrieved document
    books_list = []

    for rec in recs:
        isbn = rec.page_content.strip('"').split()[0]
        books_list.append(str(isbn))

    # Match retrieved ISBNs with the main books dataset
    book_recs = books[
        books["isbn13"].isin(books_list)
    ].head(initial_top_k)


    # --------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ].head(final_top_k)

    else:
        book_recs = book_recs.head(final_top_k)


    # --------------------------------------------------
    # EMOTION SORTING
    # --------------------------------------------------
    # Kept exactly as your original logic

    if tone == "Happy":
        book_recs.sort_values(
            by="joy",
            ascending=False,
            inplace=True
        )

    elif tone == "Surprising":
        book_recs.sort_values(
            by="surprise",
            ascending=False,
            inplace=True
        )

    elif tone == "Angry":
        book_recs.sort_values(
            by="anger",
            ascending=False,
            inplace=True
        )

    elif tone == "Suspenseful":
        book_recs.sort_values(
            by="fear",
            ascending=False,
            inplace=True
        )

    elif tone == "Sad":
        book_recs.sort_values(
            by="sadness",
            ascending=False,
            inplace=True
        )

    return book_recs


# --------------------------------------------------
# 7. FORMAT RECOMMENDATIONS FOR GRADIO
# --------------------------------------------------

def recommend_books(
        query: str,
        category: str,
        tone: str
):

    recommendations = retrieve_semantic_recommendations(
        query,
        category,
        tone
    )

    results = []

    for _, row in recommendations.iterrows():

        # Safely handle missing descriptions
        description = row["description"]

        if pd.isna(description):
            description = "No description available."

        truncated_desc_split = str(description).split()

        truncated_description = (
            " ".join(truncated_desc_split[:30])
            + "..."
        )


        # Format author names
        authors = row["authors"]

        if pd.isna(authors):
            authors_str = "Unknown author"

        else:
            authors_split = str(authors).split(";")

            if len(authors_split) == 2:
                authors_str = (
                    f"{authors_split[0]} and "
                    f"{authors_split[1]}"
                )

            elif len(authors_split) > 2:
                authors_str = (
                    f"{', '.join(authors_split[:-1])}, "
                    f"and {authors_split[-1]}"
                )

            else:
                authors_str = str(authors)


        # Create caption
        caption = (
            f"{row['title']} by {authors_str}: "
            f"{truncated_description}"
        )

        results.append(
            (
                row["large_thumbnail"],
                caption
            )
        )

    return results


# --------------------------------------------------
# 8. CATEGORY AND TONE OPTIONS
# --------------------------------------------------

categories = [
    "All"
] + sorted(
    books["simple_categories"]
    .dropna()
    .unique()
)

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad"
]


# --------------------------------------------------
# 9. GRADIO INTERFACE
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Glass()
) as dashboard:

    gr.Markdown(
        "# Semantic Book Recommender"
    )


    with gr.Row():

        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )

        submit_button = gr.Button(
            "Find recommendations"
        )


    gr.Markdown(
        "## Recommendations"
    )


    output = gr.Gallery(
        label="Recommended books",
        columns=8,
        rows=2
    )


    # Connect button to recommendation function
    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown
        ],
        outputs=output
    )


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    dashboard.launch()