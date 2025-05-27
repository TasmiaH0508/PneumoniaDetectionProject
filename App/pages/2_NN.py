import streamlit as st

from App.AppUtils import display_description

st.set_page_config(
    page_title="Predict Pneumonia with NN",
    page_icon="📊",
)

def intro():
    display_description()

intro()