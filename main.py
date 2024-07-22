import streamlit as st

from classifier import prediction

# # Dummy classifier function for demonstration
def classifier(input_text):
    return {"output": prediction(input_text)}

# # Streamlit UI
st.title("Obligation classifier")

st.write("Demo for classifiers that help lawyers parse through Obligations.")
st.write(
    str(
        "Try inputing the following:\n"
        "\"The contractor must complete the construction work by December 31st"
        " and adhere to all safety regulations throughout the project.\""
    )
)

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
