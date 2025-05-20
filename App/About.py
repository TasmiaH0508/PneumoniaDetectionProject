import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Predict Pneumonia",
    page_icon="📊",
)

def intro():
    # Goal of Project
    st.markdown(
        """
        # 🎯Goal of Project
        Train an image classifier that can distinguish x-ray photographs of patients with pneumonia 
        and those without.
        
        ### Models explored
        - SVM
        - Neural Network
        - CNN
        - Perceptron (transformed features)
        
        ### 🛠️Processing the Data

        The method of processing images for SVM and Neural Network was different from CNN.

        For SVM and Neural Network, the images where processed such that every image was represented as a list. Since 
        the images were of different sizes, padding was added for cases where the image is smaller than the target size.
        Since every x-ray photograph was 65536 pixels and there were about 3600 images, the data matrix was too 
        large, motivating the use of PCA.

        Before PCA was applied, about 10%(or 200) of the data points were randomly selected from each data group(Pneumonia
        and non-pneumonia). Normalisation was then applied to the test set. After applying PCA to the training data(and 
        reconstructing the data), features showing very little variance were removed from both the test and training data. Since
        the training set was already normalised, the test set was normalised separately on its own.
        
        The labels were then added such that if pneumonia was present, a label of 1 was given and a label of 0 to those without 
        Pneumonia.
        
        Table showing the percentage reduction in features by varying the variance of features:
        """
    )
    #todo

intro()