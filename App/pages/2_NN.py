import streamlit as st
from PIL import Image

from App.AppUtils import display_description

st.set_page_config(
    page_title="Predict Pneumonia with NN",
    page_icon="📊",
)

def intro():
    st.title("Predict Pneumonia from x-rays with Neural Network🩻")
    display_description()

    uploaded_file = st.file_uploader("Choose an image", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.uploaded_file = uploaded_file

    st.button("Predict Pneumonia",  help="Start classifying")

    display_findings()


def display_findings():
    st.markdown(
        """
        ### 🔎 About the Neural Network
        
        ###### How is this different from CNN?
        
        Instead of feeding in a 
        """
    )

intro()