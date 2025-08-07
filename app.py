import streamlit as st
import joblib
from PIL import Image

# Load model and vectorizer
model = joblib.load('language_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# UI
st.title("🌍 Language Detection App")
st.write("Enter a sentence and I will detect its language!")

# Confusion Matrix
image = Image.open('confusion_matrix.png')
st.image(image, caption='Model Confusion Matrix', use_column_width=True)

# Input Text
user_input = st.text_area("Type text here:")

if st.button("Detect Language"):
    if user_input.strip():
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)
        st.success(f"Detected Language: **{prediction[0]}**")
    else:
        st.warning("Please enter some text.")


