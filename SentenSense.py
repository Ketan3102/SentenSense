
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    model=SentenceTransformer("Ketan3101/sentensense")
    return model

@st.cache_data
def cosine_sim(sent1, sent2,_n_model):
    emb=_n_model.encode([sent1,sent2])
    cos_sim=np.dot(emb[0],emb[1])/(np.linalg.norm(emb[0])*np.linalg.norm(emb[1]))
    return cos_sim

def main():
    # Set page title and icon
    st.set_page_config(page_title="SenteSense", page_icon=":speech_balloon:")
    model=load_model()

    # Set page title
    st.title("SentenSense: Similarity Predictor")
    st.caption("Choose Text Flavor: Sentence or Paragraph")
    choice=st.selectbox('Choose One',["None","Sentence","Paragraph"],index=0)
    if choice=="Sentence":
        st.markdown("Enter your two sentences below to check their similarity!")

        # Input text
        sentence1 = st.text_input("Enter the first sentence:")
        sentence2 = st.text_input("Enter the second sentence:")

    elif choice=='Paragraph':
        st.markdown("Enter your two paragraphs below to check their similarity!")

        # Input text
        sentence1 = st.text_area("Enter the first paragraph:")
        sentence2 = st.text_area("Enter the second paragraph:")

    else:
        st.warning("Please choose a valid option.")
    
    # Button to trigger prediction
    if st.button("Predict Similarity"):
        if choice=='None':
            st.warning("Please choose one among Sentence or Paragraph.")
        elif sentence1 and sentence2:
            result=cosine_sim(sentence1,sentence2,model)

            # Display results
            st.subheader("Similarity Prediction:")
            st.write(f"Probability Score: {round(result,2)}")

            # Show overall similary
            overall_similarity = st.success("The given " + choice+'s' +" are similar.") if result> 0.90 else st.error("The given " + choice+'s' +" are not similar.")

        else:
            st.warning("Please enter both sentences to predict sentiment.")

if __name__=="__main__":
    main()
