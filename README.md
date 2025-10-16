 MediGPT: Extractive Health Chatbot
<img width="1149" height="548" alt="Skermskoot 2025-10-16 205302" src="https://github.com/user-attachments/assets/db05afd8-4d90-4818-adff-6ad0be9e83a3" />

MediGPT is a specialized, domain-specific Question Answering (QA) system designed to provide concise, fact-based answers by extracting information directly from a curated medical corpus. Built using a fine-tuned DistilBERT Transformer model, this application emphasizes safety, relevance, and data reliability through multiple layers of intelligent validation.

Core Features & Technical Highlights

This project addresses the critical need for reliable information in specialized domains by incorporating advanced NLP techniques:

1. Extractive Question Answering (QA)

Model: Utilizes a fine-tuned DistilBERT Transformer model implemented in TensorFlow.

Functionality: Instead of generating free-form text, the model extracts the most probable answer span from a retrieved medical context, ensuring responses are directly supported by the source material.

2. Specialized Domain Guardrail (Crucial for Rubric Compliance)

A TF-IDF Vectorizer and Cosine Similarity check is implemented as a pre-processing guardrail.

This mechanism filters out out-of-domain queries (e.g., questions about finance or technology) with a threshold of 0.08, preventing the Transformer from running and returning the non-medical answer, thereby enforcing strict domain specialization.

3. Confidence Scoring

All successful answers are accompanied by a confidence score, derived from the aggregated start/end token logits. This metric provides a crucial indicator of answer reliability, flagging ambiguous or low-confidence extractions to the user.

4. Large File Storage (Git LFS)

The large model weights file (tf_model.h5, ~253MB) is managed using Git Large File Storage (LFS) to overcome GitHub's 100MB file limit.

 Getting Started

Prerequisites

To run this application, you must have the following dependencies installed. They are listed in requirements.txt.

Python 3.13+ (or environment compatible with TensorFlow 2.20.0)

streamlit

pandas

transformers

tensorflow==2.20.0

scikit-learn

Local Setup

Clone the repository:

git clone [https://github.com/nmaketh/MediGPT-HealthChatbot.git](https://github.com/nmaketh/MediGPT-HealthChatbot.git)
cd MediGPT-HealthChatbot



Ensure Git LFS is installed (essential for downloading tf_model.h5):

git lfs install
git lfs fetch



Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`



Install the required libraries:

pip install -r requirements.txt



Run the Streamlit application:

streamlit run streamlit_app.py



The application will open in your default web browser (usually at http://localhost:8501).
STREAMLIT DEPLOYMENT SUCCESSFULLY:(https://medigpt-healthchatbot-mzvzqubj9b5kmxzckjdwzj.streamlit.app/)

🎥 Demonstration Video

To see the MediGPT system in action, including the Domain Guardrail and Confidence Scoring features, please watch the demonstration video:

Video Link: https://youtu.be/47Mf2PpC21k

 Demonstration & Usage

The application is designed to handle three distinct types of queries:

Test Case

Example Query

Expected Outcome

Rationale

Success Case

What are the symptoms of food poisoning?

Clear Extracted Answer with a High Confidence Score (> 0.50).

Query is highly relevant, allowing the DistilBERT model to perform high-fidelity extraction.

Out-of-Domain Failure

How do I upgrade my computer's RAM?

Immediate Rejection Message and a Confidence Score of 0.00.

Fails the TF-IDF domain guardrail, proving specialization and preventing irrelevant responses.

Low Confidence Check

Tell me about eyes.

Low Confidence Message (e.g., Score < 0.20) asking the user to rephrase.

Passes the domain check but fails the internal confidence threshold, indicating a vague query with no single, specific answer.

📁 Project Structure

File/Directory

Description

streamlit_app.py

The main application file containing the Streamlit UI, TF-IDF guardrail logic, and the core DistilBERT QA function.

requirements.txt

Lists all necessary Python dependencies, including tensorflow==2.20.0 for deployment compatibility.

medquad.csv

The curated dataset used for both the TF-IDF vectorizer and the context retrieval pool.

fine_tuned_qa_model/

Directory containing the fine-tuned DistilBERT model weights (tf_model.h5) and configuration files.

.gitattributes

Configuration file that tells Git to use LFS for the tf_model.h5 file.
