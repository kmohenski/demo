from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import streamlit as st
import torch

hf_token = st.secrets["huggingface_token"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("lejzi/obligation_classifier", token=hf_token)
loaded = AutoModelForSequenceClassification.from_pretrained("lejzi/obligation_classifier", token=hf_token)
loaded.to(device)

pipe = TextClassificationPipeline(model=loaded, tokenizer=tokenizer, return_all_scores=True)

def classify_obligations(sentence: str) -> str:
    result = pipe(sentence)
    best = {"label": '', "score": -1.0}

    for pair in result[0]:
        if pair["score"] >= best["score"]:
            best["label"] = pair["label"]
            best["score"] = pair["score"]

    output = best["label"]

    return output
