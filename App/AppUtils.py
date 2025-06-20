import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import torch
from PIL import Image

from App.ProcessRawData import min_max_normalise_with_predefined_params, pad_image


def display_description():
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
    st.warning("⚠️ Use the models at your own risk. The models are not 100% accurate.")
    st.subheader("⬇️ Upload your chest x-ray here:")

def create_line_chart(data, labels, index, x_axis_label, domain):
    # Prepare data
    df = pd.DataFrame(data, columns=labels, index=index)
    df[x_axis_label] = df.index
    df = df.reset_index(drop=True)

    # Melt DataFrame to long format for Altair
    df_long = df.melt(id_vars=[x_axis_label], var_name="Metric", value_name="Score")

    # Create Altair chart with controlled y-axis scale
    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X(f"{x_axis_label}:Q", title=x_axis_label),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=domain)),
        color="Metric:N"
    ).properties(
        width=700,
        height=400,
    )

    st.altair_chart(chart, use_container_width=True)

def create_unstacked_bar_graph(data, labels, index, x_axis_label):
    chart_data = pd.DataFrame(
        data,
        columns=labels,
        index=index
    )
    st.bar_chart(chart_data, x_label=x_axis_label, stack=False)

def process_image(image_path):
    '''
    Returns image as an array after it has been processed for NN and SVM
    :param image_path: path to image
    :return: image as an array
    '''
    image = Image.open(image_path).convert('L')
    padded_and_cropped_image = pad_image(image)
    image_array = np.array(padded_and_cropped_image).flatten()
    # need the scaling
    image_array = image_array / 255
    image_array = np.reshape(image_array, (1, image_array.shape[0]))
    min_matrix = np.load("App/Models/Data/ProcessedRawData/MinData/min_across_all_features.npy")
    min_matrix = np.reshape(min_matrix, (image_array.shape[0], image_array.shape[1]))
    range_matrix = np.load("App/Models/Data/ProcessedRawData/RangeData/range_across_all_features.npy")
    range_matrix = np.reshape(range_matrix, (image_array.shape[0], image_array.shape[1]))
    normalised_arr = min_max_normalise_with_predefined_params(image_array, min_matrix, range_matrix)
    bias = np.ones((1, 1))
    normalised_arr = np.hstack((bias, normalised_arr))
    normalised_arr = torch.from_numpy(normalised_arr)
    return normalised_arr