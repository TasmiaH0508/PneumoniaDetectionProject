import streamlit as st
from PIL import Image

from App.AppUtils import display_description
from App.Models.CNN.CNN import predict_with_saved_model

st.set_page_config(
    page_title="Predict Pneumonia with CNN",
    page_icon="📊",
)

def intro():
    st.title("Predict Pneumonia from x-rays with Convolutional Neural Network🩻")

    display_description()

    uploaded_file = st.file_uploader("Choose an image", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.uploaded_file = image

    st.button("Predict Pneumonia", on_click=predict, help="Start classifying")

    if st.session_state.get("no_file_uploaded_error", False):
        st.error("Please upload an image.")
    elif st.session_state.get("is_predicted", False):
        if st.session_state.get("prediction") == 0.0:
            st.success(
                """
                ### 📊Evaluation results
                🟢 No pneumonia was detected.
                """
            )
        else:
            st.error(
                """
                ### 📊Evaluation results
                🔴 Pneumonia was detected.
                """
            )
    else:
        st.warning("Please ensure that you have uploaded an image.")

def predict():
    if st.session_state.uploaded_file is None:
        st.session_state.no_file_uploaded_error = True
        st.session_state.is_predicted = False
    else:
        st.session_state.no_file_uploaded_error = False
        image = st.session_state.uploaded_file
        pred = predict_with_saved_model(image)
        st.session_state.prediction = pred
        st.session_state.is_predicted = True

def display_findings():
    #todo
    return 0

intro()