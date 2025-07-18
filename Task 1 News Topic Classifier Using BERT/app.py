import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./ag_news_bert')
    tokenizer = BertTokenizerFast.from_pretrained('./ag_news_bert')
    return model, tokenizer

model, tokenizer = load_model()

# Class labels (AG News categories)
class_labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Streamlit app
st.title("News Topic Classifier")
st.write("Classify news headlines into one of four categories: World, Sports, Business, or Sci/Tech")

user_input = st.text_area("Enter news headline or text:", "")

if st.button("Classify"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Display result
        st.success(f"Predicted category: {class_labels[predicted_class]}")
    else:
        st.warning("Please enter some text to classify.")