import streamlit as st
from PIL import Image

from App.Analysis.PredictPneumonia import predict_pneumonia

st.set_page_config(
    page_title="Predict Pneumonia",
    page_icon="📊",
)

def intro():
    # Project description
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
        """
    )
    st.warning("⚠️ Use the models at your own discretion.")
    st.subheader("⬇️ Upload your chest x-ray here:")

    # for image upload
    uploaded_file = st.file_uploader("Choose an image", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.image_given = True
        st.session_state.image = uploaded_file

    st.button("Predict Pneumonia", help="Start classifying")

    # error message if no image is given
    if st.session_state.get("no_image_given_error", False):
        st.error("No image given.")

    st.info(
        """
        What model have we used here?
        
        Here, we have used SVM
        """
    )

    st.sidebar.success("ML Models")

intro()