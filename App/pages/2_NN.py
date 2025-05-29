import streamlit as st
from PIL import Image

from App.AppUtils import display_description

st.set_page_config(
    page_title="Predict Pneumonia with NN",
    page_icon="📊",
)

# need to vary lr, feature variance, epochs, model
# metrics: accuracy, precision, recall

def intro():
    st.title("Predict Pneumonia from x-rays with Neural Network🩻")
    display_description()

    uploaded_file = st.file_uploader("Choose an image", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.uploaded_file = uploaded_file

    st.button("Predict Pneumonia",  help="Start classifying")

intro()