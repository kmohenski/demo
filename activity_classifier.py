from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import streamlit as st
import torch

hf_token = st.secrets["huggingface_token"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("lejzi/activity_classifier", token=hf_token)
loaded = AutoModelForSequenceClassification.from_pretrained("lejzi/activity_classifier", token=hf_token)
loaded.to(device)

pipe = TextClassificationPipeline(model=loaded, tokenizer=tokenizer, return_all_scores=True)

def classify_activities(obligation_text: str) -> str:
    result = pipe(obligation_text)
    output = []

    for pair in result[0]:
        if pair["score"] >= 0.5:
            output.append(pair["label"])

    return output
