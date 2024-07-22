import streamlit as st

from classifier import prediction

# # Dummy classifier function for demonstration
def classifier(input_text):
    return {"output": prediction(input_text)}

# # Streamlit UI
st.title("Obligation classifier")

# # Text input
input_text = st.text_input("Input Text")

# # Classify button
if st.button("Classify"):
    if input_text:
        # Display the result
        st.write("Classification Result:")
        st.json(classifier(input_text))
    else:
        st.write("Please enter text to classify.")
