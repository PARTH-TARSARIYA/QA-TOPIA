import streamlit as st
import requests

st.set_page_config(page_title="Topic Modeling QA System", layout="wide")

st.title("📊 Topic Modeling QA Dashboard")
st.markdown("Enter a question and answer to classify topic and retrieve similar insights.")

col1, col2 = st.columns(2)

with col1:
    user_question = st.text_area("Enter Question")

with col2:
    user_answer = st.text_area("Enter Answer")

if st.button("Analyze"):

    if user_question.strip() == "" or user_answer.strip() == "":
        st.warning("Please enter both question and answer.")
    else:

        response = requests.post(
            "http://localhost:8000/analyze",
            json={
                "question": user_question,
                "answer": user_answer
            }
        )

        data = response.json()

        topic = data["topic"]
        confidence = data["confidence"]
        related_questions = data["questions"]
        related_answers = data["answers"]

        if related_answers == [] or related_questions == [] or confidence < 0.7:
            use_similarity = False
            st.write("Enter related to research question and answer!")
        else:
            use_similarity = True
            st.subheader('Related Topic')
            st.success(f"{topic}  \nConfidence : {round(confidence, 4)}")

        if use_similarity:
            st.subheader("📚 Similar Answers")

            for sim, ans, t in related_answers:
                st.markdown(f"""
                **Answer :** {ans}  
                Topic: `{t}`  
                Similarity: `{sim}`
                """)
                st.markdown("---")

        st.sidebar.title("🔍 Related Questions")

        if use_similarity:
            for sim, que, t in related_questions:
                st.sidebar.markdown(f"""
                **Question :** {que}  
                Topic: `{t}`  
                Similarity: `{sim}`
                """)
                st.sidebar.markdown("---")
        else:
            st.sidebar.info("Enter related to research question and answer!")
