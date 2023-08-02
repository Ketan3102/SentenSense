import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

def main():
    def cosine_sim(sent1, sent2,n_model):
        emb=n_model.encode([sent1,sent2])
        cos_sim=np.dot(emb[0],emb[1])/(np.linalg.norm(emb[0])*np.linalg.norm(emb[1]))
        return cos_sim

    # Set page title and icon
    st.set_page_config(page_title="SenteSense", page_icon=":sparkles:")

    # Set page header
    st.title("SentenSense: Similarity Predictor")
    st.markdown("Enter two sentences below to check their similarity!")

    # Input text
    sentence1 = st.text_input("Enter the first sentence:")
    sentence2 = st.text_input("Enter the second sentence:")

    # Sentiment analysis model
    model=SentenceTransformer("Ketan3101/sentensense")

    # Button to trigger prediction
    if st.button("Predict Sentiment"):
        if sentence1 and sentence2:
            result=cosine_sim(sentence1,sentence2,model)

            # Display results
            st.subheader("Similarity Prediction:")
            st.write(f"1. Sentence: '{sentence1}'")
            st.write(f"2. Sentence: '{sentence2}'")
            st.write(f"probability score: {round(result,2)})")

            # Show overall sentiment
            overall_sentiment = "The given Sentences are similarity" if result> 0.90 else "The given Sentences are not similar."
            st.subheader(f"Overall Analysis: {overall_sentiment}")

        else:
            st.warning("Please enter both sentences to predict sentiment.")
if __name__=="__main__":
    main()
