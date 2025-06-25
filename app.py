import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Load data
df = pd.read_csv("cleaned_FQA.csv")
df.columns = df.columns.str.strip()

documents = df["question"].tolist()
answers = df["answer"].tolist()

# Fit TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

translator = Translator()

# Answer function
def answer_question(question, top_k=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = {
            "question": documents[idx],
            "similarity_score": round(similarities[idx], 3),
            "answer": answers[idx]
        }
        results.append(result)
    return results

def translate_answer(answer, target_lang):
    if target_lang == "en":
        return translator.translate(answer, src="ms", dest="en").text
    elif target_lang == "ms":
        return translator.translate(answer, src="en", dest="ms").text
    return answer

# Streamlit UI
st.title("Malaysian Tourism QA Web App(Malay or English)")
st.markdown("Ask a question in English or Malay.")

lang_choice = st.radio("Choose your language:", ["ms", "en"], format_func=lambda x: "Malay" if x == "ms" else "English")
user_question = st.text_area("Enter your question here:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        results = answer_question(user_question, top_k=3)
        for i, r in enumerate(results, 1):
            st.markdown(f"### Result #{i}")
            st.write(f"**Similarity Score:** {r['similarity_score']}")
            if lang_choice == "en":
                translated = translate_answer(r["answer"], "en")
                st.write(f"**Answer (English):** {translated}")
                st.caption(f"Original (Malay): {r['answer']}")
            else:
                st.write(f"**Answer (Malay):** {r['answer']}")