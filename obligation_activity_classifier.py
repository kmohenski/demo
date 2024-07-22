from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import streamlit as st
import torch

hf_token = st.secrets["huggingface_token"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("lejzi/activity_classifier", hf_token)
loaded = AutoModelForSequenceClassification.from_pretrained("lejzi/activity_classifier", hf_token)
loaded.to(device)

pipe = TextClassificationPipeline(model=loaded, tokenizer=tokenizer, return_all_scores=True)

def classify_obligation_activities(obligation_text: str) -> str:
    """Classify the activities contained in the sentance. The sentence is classified
    by a classifier stored in `activity_classifier` directory.
    There can be any number of activities in a sentence so we return any which score over 0.5.

    :param obligation_text: sentence to be classified
    :return: a string indicating the type of obligation

    Example:
    ```
        [
            'Organizing and conducting external meetings',
            'Submitting reports/notifications',
            'Signing/notarising'
        ]
    ```
    """
    result = pipe(obligation_text)
    output = []

    for pair in result[0]:
        if pair["score"] >= 0.5:
            output.append(pair["label"])

    return output
