import streamlit as st

def display_description():
    st.title("Predict Pneumonia from x-rays 🩻")
    st.markdown(
        "In this project, several models(NN, SVM and CNN) were trained to predict pneumonia from chest x-ray images. "
        "Feel free to use the best classifier trained so far below. Afterwards, you may experiment with the other "
        "models in the dropdown on the left."
    )
    st.info(
        """
        ### ⭕️ Guidelines for images:
        - **Only chest x-ray images** can be used.
        - For best results, ensure that the chest x-ray images are square or close to being a square. 
        """
    )
    st.warning("⚠️ Use the models at your own discretion.")
    st.subheader("⬇️ Upload your chest x-ray here:")