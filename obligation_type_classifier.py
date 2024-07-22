from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import streamlit as st
import torch

hf_token = st.secrets["huggingface_token"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("lejzi/category_classifier", token=hf_token)
loaded = AutoModelForSequenceClassification.from_pretrained("lejzi/category_classifier", token=hf_token)
loaded.to(device)

pipe = TextClassificationPipeline(model=loaded, tokenizer=tokenizer, return_all_scores=True)

def classify_obligation_type(obligation_text: str) -> str:
    """Classify the sentences into categories.
    There is only ever one type so we only return the best scoring type.

    :param obligation_text: sentence to be classified
    :return: a string indicating the type of obligation

    Example:
    ```
        'Obligation to report hazards'
    ```
    """
    result = pipe(obligation_text)
    best = {"label": '', "score": -1.0}

    for pair in result[0]:
        if pair["score"] >= best["score"]:
            best["label"] = pair["label"]
            best["score"] = pair["score"]

    return best["label"]
