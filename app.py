import json
import re

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------
# NLTK downloads
# ---------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ---------------------------------------
# Load FAQ data
# ---------------------------------------

with open("faq_data.json", "r", encoding="utf-8") as file:

    faq_data = json.load(file)


# ---------------------------------------
# NLP preprocessing
# ---------------------------------------

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Split into words
    tokens = text.split()

    processed_tokens = []

    for token in tokens:

        # Remove stopwords
        if token not in stop_words:

            # Lemmatization
            token = lemmatizer.lemmatize(token)

            processed_tokens.append(token)

    return " ".join(processed_tokens)


# ---------------------------------------
# Prepare FAQ questions
# ---------------------------------------

faq_questions = [
    item["question"]
    for item in faq_data
]

faq_answers = [
    item["answer"]
    for item in faq_data
]


processed_questions = [
    preprocess_text(question)
    for question in faq_questions
]


# ---------------------------------------
# TF-IDF
# ---------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)


faq_vectors = vectorizer.fit_transform(
    processed_questions
)


# ---------------------------------------
# Find answer
# ---------------------------------------

def get_answer(
    user_question,
    threshold=0.25
):

    # Preprocess user question
    processed_query = preprocess_text(
        user_question
    )

    # Convert query to TF-IDF vector
    query_vector = vectorizer.transform(
        [processed_query]
    )

    # Calculate similarity
    similarities = cosine_similarity(
        query_vector,
        faq_vectors
    )[0]

    # Find best match
    best_index = similarities.argmax()

    # Get similarity score
    best_score = similarities[best_index]

    # If score is too low
    if best_score < threshold:

        return (
            "Sorry, I don't have an answer "
            "for that question. Please try "
            "asking something related to the FAQs."
        )

    # Return matching answer
    return faq_answers[best_index]


# ---------------------------------------
# Streamlit UI
# ---------------------------------------

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)


st.title("🤖 FAQ Chatbot")

st.write(
    "Ask a question and I will try to "
    "find the most relevant answer."
)


# ---------------------------------------
# Chat history
# ---------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------
# Display previous messages
# ---------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ---------------------------------------
# User input
# ---------------------------------------

user_question = st.chat_input(
    "Type your question here..."
)


if user_question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.write(user_question)


    # Get chatbot answer
    answer = get_answer(
        user_question
    )


    # Add bot message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # Display bot answer
    with st.chat_message(
        "assistant"
    ):

        st.write(answer)