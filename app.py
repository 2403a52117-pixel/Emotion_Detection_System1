import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return labels[torch.argmax(logits).item()]

st.title("Emotion Detection using DistilBERT")
text = st.text_area("Enter text")

if st.button("Predict"):
    if text.strip():
        st.success(predict(text))
    else:
        st.warning("Enter text first")
