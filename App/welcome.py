import streamlit as st
from PIL import Image

def intro():
    st.title("Predict Pneumonia from x-rays 🩻")
    st.markdown(
        "In this project, several models(NN, SVM and CNN) were trained to predict pneumonia from chest x-ray images. "
        "Feel free to use the best classifier trained so far below. Afterwards, you may experiment with the other "
        "models in the dropdown on the left."
    )
    st.subheader("⭕️ Guidelines for images:")
    st.markdown(
        """
        - **Only chest x-ray images** can be used.
        - For best results, ensure that the chest x-ray images are **256 by 256 pixels** of the most appropriate regions. 
        
        **⚠️ Use the models at your own discretion.**
        """
    )
    st.subheader("⬇️ Upload your chest x-ray here:")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    st.button("Predict Pneumonia", on_click=predict, help="Start classifying")

    if st.session_state.get("predicted", False):
        st.markdown("---")  # Horizontal line for separation
        st.subheader("📊 Prediction Results")
        st.success("✅ Pneumonia detected with 94% confidence")  # Example output

    st.sidebar.success("ML Models")

def predict():
    st.session_state.predicted = True

intro()