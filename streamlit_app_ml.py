# ============================================================
# 🧠 MediGPT: Extractive Medical Chatbot (DistilBERT)
# ============================================================
# This script implements an Extractive Question Answering system
# using the DistilBERT model, which extracts the answer span from context.
# It includes a robust TF-IDF guardrail to ensure the chatbot 
# remains specialized to the medical domain, as required by the rubric.

import os
import streamlit as st
import pandas as pd
import random
import re
from typing import Tuple
import numpy as np

# --- Configuration for TensorFlow and Tokenizers ---
# Fix for Keras 3 incompatibility with Hugging Face TF models
os.environ["TF_USE_DEEP_LEARNING_FLAG"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Attempt to load necessary libraries ---
try:
    from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics.pairwise import cosine_similarity
    import tensorflow as tf
except ImportError:
    st.error("Missing required libraries. Please ensure you have installed: `transformers`, `tensorflow`, `pandas`, and `scikit-learn`.")
    st.stop()


# ============================================================
# 1. MODEL AND DATA LOADING (Cached Resources)
# ============================================================
QA_MODEL_NAME = "distilbert-base-uncased" # Base model used for Extractive QA
DOMAIN_THRESHOLD = 0.08 # Final threshold to block non-medical queries
MIN_Q_LENGTH = 3        # Minimum words in a meaningful question
CONTEXT_POOL_SIZE = 50  # Number of contexts to check for the best answer

@st.cache_resource(show_spinner="Loading DistilBERT Extractive Model...")
def load_qa_model() -> Tuple[AutoTokenizer, TFAutoModelForQuestionAnswering]:
    """Loads the pre-trained DistilBERT model and tokenizer for QA."""
    try:
        # Load the DistilBERT model for Question Answering (TensorFlow version)
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        # NOTE: We explicitly set use_safetensors=False to avoid 'safe_open' errors 
        # that sometimes occur in specific environments when loading model weights.
        model = TFAutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME, use_safetensors=False)
        st.success(f"DistilBERT Extractive Model ({QA_MODEL_NAME}) loaded successfully.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading QA Model: {e}")
        st.warning("Could not load the Extractive QA model.")
        st.stop()

@st.cache_data(show_spinner="Loading Medical Context Data and Vectorizing...")
def load_medical_data() -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """Loads medical data and fits the TF-IDF Vectorizer for domain filtering."""
    try:
        # Load the uploaded dataset
        df = pd.read_csv("medquad.csv")
        df = df[['question', 'answer']].dropna().reset_index(drop=True)
        # Combine Q and A to create a rich context corpus for better filtering
        df['context'] = df['question'] + " " + df['answer']
        
        # Fit the vectorizer on a large sample of the corpus
        corpus = df['context'].sample(min(2000, len(df)), random_state=42).tolist()
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), # Use bigrams for better contextual keyword matching
            max_df=0.8
        ).fit(corpus)
        
        return df, vectorizer
    except Exception as e:
        st.error(f"Error loading and vectorizing data: {e}")
        st.stop()

# Initialize resources
tokenizer, model = load_qa_model()
df, vectorizer = load_medical_data()


# ============================================================
# 2. DOMAIN GUARDRAIL AND CONTEXT RETRIEVAL
# ============================================================
def is_in_domain(question: str) -> bool:
    """
    Uses TF-IDF and Cosine Similarity to determine if a question is health-related.
    """
    # Clean the question
    q_clean = re.sub(r'[^\w\s]', '', question.lower())
    q_words = [word for word in q_clean.split() if word not in ENGLISH_STOP_WORDS]
    
    # Check minimum length
    if len(q_words) < MIN_Q_LENGTH:
        return False

    # Calculate similarity against the entire medical corpus
    q_vec = vectorizer.transform([" ".join(q_words)])
    corpus_vecs = vectorizer.transform(df['context'].tolist())
    similarity_scores = cosine_similarity(q_vec, corpus_vecs)[0]
    
    max_similarity = similarity_scores.max()

    return max_similarity > DOMAIN_THRESHOLD


def retrieve_context_set(question: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves a set of the most relevant contexts using TF-IDF and Cosine Similarity.
    """
    q_vec = vectorizer.transform([question])
    
    # Compare against the rich context column
    context_vecs = vectorizer.transform(df['context'].tolist())
    similarity_scores = cosine_similarity(q_vec, context_vecs)[0]
    
    # Get indices of the top contexts
    top_indices = np.argsort(similarity_scores)[-CONTEXT_POOL_SIZE:]
    
    # Return the corresponding rows (context and score)
    return df.iloc[top_indices].assign(score=similarity_scores[top_indices])


# ============================================================
# 3. CORE EXTRACTIVE QA FUNCTION
# ============================================================
def get_answer(question: str) -> str:
    """
    Main function to retrieve context and extract the best answer using DistilBERT.
    """
    
    # Guardrail 1: Domain Check (Critical for the rubric)
    if not is_in_domain(question):
        return (
            "❌ I am a specialized medical chatbot. Please ask a health or disease-related question."
            "\n\n**Confidence Score:** 0.00\n\n**Context Used:** N/A (Query rejected by domain filter)"
        )

    # Step 1: Context Retrieval
    context_set = retrieve_context_set(question, df)
    
    best_answer = "Sorry, I could not find a meaningful answer in the medical context."
    best_total_score = -1e9 # Initialize to a very low value
    final_context = ""
    
    # Step 2: Iterate and Extract
    for _, row in context_set.iterrows():
        context = row['answer']
        retrieval_score = row['score']

        # Tokenize the question-context pair
        inputs = tokenizer(question, context, return_tensors="tf", truncation=True, padding='max_length', max_length=512)
        
        # Get the start and end logits (scores)
        outputs = model(inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Find the indices with the maximum scores
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0]

        # Get the tokenized output
        tokens = inputs.input_ids.numpy()[0]
        
        # Extractive QA confidence is often sum/max of start/end logits
        # We use a combined score and boost it by the retrieval score
        qa_score = start_logits[0, start_index].numpy() + end_logits[0, end_index].numpy()
        total_score = qa_score + retrieval_score

        # Convert tokens back to text span
        if end_index >= start_index:
            answer_tokens = tokens[start_index:end_index + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Update best answer if the current score is higher AND the answer is meaningful
            if total_score > best_total_score and answer.lower() not in ('[cls]', 'unk', ''):
                best_total_score = total_score
                best_answer = answer
                final_context = context
                
    # Step 3: Final Answer Guardrails (Confidence Check)
    
    # Normalize score (logits are typically high, so divide to make it manageable)
    normalized_score = max(0.0, best_total_score / 10.0) 

    if normalized_score < 0.2:
        return (
            "The model found information but the **confidence score was too low**. Please rephrase your question or be more specific."
            f"\n\n🔍 **Confidence Score:** {normalized_score:.2f}"
            f"\n\n**Context Used:** {final_context[:150]}..."
        )

    # Combine the final output with metadata
    return (
        f"{best_answer}\n\n"
        f"🔍 **Confidence Score:** {normalized_score:.2f}\n\n"
        f"**Context Used:** {final_context[:150]}..."
    )


# ============================================================
# 4. STREAMLIT INTERFACE (UI Integration)
# ============================================================
st.set_page_config(page_title="MediGPT - Extractive Chatbot", page_icon="💊", layout="centered")

st.markdown(
    """
    <style>
    /* Styling for the Extractive DistilBERT version */
    .stButton>button {
        width: 100%;
        background-color: #004d40; /* Darker Teal theme */
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00695c;
    }
    .stSuccess {
        background-color: #e0f2f1;
        border-left: 5px solid #004d40;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 MediGPT: Extractive Medical Chatbot (DistilBERT)")
st.subheader("Extracts the best answer directly from medical context.")

st.markdown("---")

user_question = st.text_input(
    "🩺 **Ask a specific medical question:**",
    placeholder="e.g., What are the common symptoms of influenza?",
    key="user_q"
)

# Create an empty placeholder for the response area (ensures previous results are cleared)
response_placeholder = st.empty()

if st.button("Get Extractive Answer"):
    if user_question.strip():
        with st.spinner("Analyzing question with Transformer Model..."):
            response = get_answer(user_question)
        
        # Update the placeholder content
        with response_placeholder.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("🤖 **Chatbot Response**")
            st.markdown(response)
    else:
        # Clear the placeholder and show a warning
        response_placeholder.empty()
        st.warning("Please enter your question before asking MediGPT.")

st.markdown("---")
st.caption(f"Model: {QA_MODEL_NAME} | Extractive QA Approach | **Specialized Domain Filter**")
