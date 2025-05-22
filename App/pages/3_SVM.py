import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

st.set_page_config(
    page_title="Predict Pneumonia with SVM",
    page_icon="📊",
)

# for linear kernel, for data with no features removed
results_1 = np.array([[0.91131, 0.91131, 0.91131],
                    [0.89, 0.89, 0.89],
                    [0.94, 0.94, 0.94]])
results_1 = np.transpose(results_1)
results_1_labels = ["accuracy", "precision", "recall"]
results_1_index = [1, 2, 3]

# for gaussian kernel, varying gamma, with no features removed
results_2 = np.array([[0.931, 0.932, 0.930, 0.924, 0.923, 0.923],
                      [0.918, 0.920, 0.919, 0.911, 0.912, 0.903],
                      [0.948, 0.946, 0.944, 0.940, 0.937, 0.9475]])
results_2 = np.transpose(results_2)
results_2_labels = results_1_labels
results_2_index = [1/2500, 1/3000, 1/4000, 1/6000, 1/8000, 1/1000]

# for linear kernel, for data with features removed
results_3 = np.array([[0.909, 0.909, 0.909],
                      [0.892, 0.892, 0.892],
                      [0.931, 0.931, 0.931]])
results_3 = np.transpose(results_3)
results_3_labels = results_1_labels
results_3_index = [1, 2, 3]

# for gaussian kernel, for data with features removed
results_4 = np.array([[0.930, 0.929, 0.928, 0.923, 0.922, 0.924],
                      [0.920, 0.920, 0.921, 0.915, 0.913, 0.909],
                      [0.941, 0.940, 0.936, 0.934, 0.932, 0.942]])
results_4 = np.transpose(results_4)
results_4_labels = ["accuracy", "precision", "recall"]
results_4_index = results_2_index

def intro():
    # Project description
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

    st.button("Predict Pneumonia", on_click=predict, help="Start classifying")

    # error message if no image is given
    if st.session_state.get("no_image_given_error", False):
        st.error("No image given.")

    # for displaying the predictions
    if st.session_state.get("predicted", False):
        if st.session_state.get("pneumonia_not_detected", False):
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            st.success("🟢 Pneumonia not detected.")
            st.markdown("---")
        elif st.session_state.get("pneumonia_detected", False):
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            st.error("🔴 Pneumonia detected.")
            st.markdown("---")
        else:
            st.warning("Evaluation failed.")

    st.markdown(
        """
        ### 🔎 About the SVM model
        
        The classifier was trained on 2 kinds of data: 
        - **Dataset A:** The first data set was created by converting each images into a vector. No features were removed.
        - **Dataset B:** The second data set was created by converting each images into a vector and removing features based on the 
        threshold variance(i.e. if the variance of the feature a feature across all samples is lower than the threshold 
        variance, the feature is removed). A feature-mapping was performed based on the training data set so that the 
        test set can be manipulated in the same way.
        
        ###### Fig 1: Bar graph showing how accuracy, precision and recall varies with polynomial degree for SVM 
        ###### classifier(linear) trained on Dataset A
        """
    )

    # display results for varying polynomial degree, using linear kernel, for data with no features removed.
    create_unstacked_bar_graph(results_1, results_1_labels, results_1_index, "Polynomial degree")
    st.markdown(
        """
        From the bar graph above, data (in dataset A) appears to be linearly separable.
        
        ###### Fig 2: Line graph showing how accuracy, precision and recall varies with gamma for SVM 
        ###### classifier(Gaussian) trained on Dataset A
        """
    )

    create_line_chart(results_2, results_2_labels, results_2_index, "Gamma")

    st.markdown(
        """
        ###### Fig 3: Bar graph showing how accuracy, precision and recall varies with polynomial degree for SVM 
        ###### classifier(linear) trained on Dataset B
        """
    )
    # linear kernel, dataset B
    create_unstacked_bar_graph(results_3, results_3_labels, results_3_index, "Polynomial degree")
    st.markdown(
        """
        Clearly, for Dataset B, from Fig 3 the data is linearly separable.
        
        ###### Fig 4: Line graph showing how accuracy, precision and recall varies with gamma for SVM 
        ###### classifier(Gaussian) trained on Dataset B
        """
    )
    # gaussian kernel, dataset B
    create_line_chart(results_4, results_4_labels, results_4_index, "Gamma")


def create_unstacked_bar_graph(data, labels, index, x_axis_label):
    chart_data = pd.DataFrame(
        data,
        columns=labels,
        index=index
    )
    st.bar_chart(chart_data, x_label=x_axis_label, stack=False)

def create_line_chart(data, labels, index, x_axis_label):
    # Prepare data
    df = pd.DataFrame(data, columns=labels, index=index)
    df[x_axis_label] = df.index
    df = df.reset_index(drop=True)

    # Melt DataFrame to long format for Altair
    df_long = df.melt(id_vars=[x_axis_label], var_name="Metric", value_name="Score")

    # Create Altair chart with controlled y-axis scale
    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X(f"{x_axis_label}:Q", title=x_axis_label),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.9, 0.95])),
        color="Metric:N"
    ).properties(
        width=700,
        height=400,
    )

    st.altair_chart(chart, use_container_width=True)


def predict():
    if not (st.session_state.get("image_given", False)):
        st.session_state.no_image_given_error = True
    else:
        st.session_state.predicted = True
        st.session_state.no_image_given_error = False


intro()