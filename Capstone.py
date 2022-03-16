# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:12:49 2022

@author: David
"""
# importing necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean
import streamlit as st 
from PIL import Image
fig = plt.figure()
fig1= plt.figure()

#with open("customs.css") as f:
    #st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
      
st.title('Concrete Classifier using mean pixel and image centering image processing techniques')

   
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:    
        img = Image.open(file_uploaded)
        st.text('Original Image')
        st.image(img, caption='Uploaded Image', use_column_width=True)

    img_new = np.array(img.convert('RGB'))
    IMG_SIZE = 225
    img=cv2.resize(img_new, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image_dimension = gray.shape

    st.write("Dimensions of original image")
    st.write(image_dimension)

    st.write("Average pixel intensity before normalization")
    Average1_img = img.mean()
    st.write(Average1_img)

    st.write("Normalizing Image .......")
    #Normalize crack image
    gray_array = gray.astype(np.float64)
    img_normalized = gray_array/255.0

    plt.hist(img_normalized)
    st.text('Distribution of pixel intensities')
    st.pyplot(fig)

    st.write("The mean of normalized image is: ")
    
    #Perform pixel subtraction across all images
    img_mean = img_normalized.mean()
    st.write(img_mean)


    st.header("Approach 1:")
    st.header("Experimentation to determine threshold if average mean is used as criteria for classificatio. Threshold selected from several iterations is 0.75 (cracked image is less than 0.75 and no crack is greater than 0.75")
    #Threshold testing and use in classification based on average pixel Intensity
    if img_mean < 0.75:
      st.write("Prediction: This concrete has a crack")
    else:
      st.write("Prediction: This concrete has no crack")
    plt.imshow(img_normalized,cmap='gray')
    st.text('Predicted Image')
    st.pyplot(fig)
    
    st.header("Approach 2:")
    st.header("Classifying crack versus no crack based on pixel centering: Each pixel value is subtracted from mean pixel value to determine if image is a crack or not")
    
    #Normalize crack image
    gray_array = gray.astype(np.float64)
    img_normalized = gray_array/255.0
    
    #Subtract pixels
    img_mean = img_normalized.mean()
    img_centered = img_normalized - img_mean
    max_centered = img_centered.max()
    
    st.write("The max of centered image is: ")
    
    st.write(max_centered)
    fig1 = plt.hist(max_centered)
    st.text('Distribution of max centered pixel intensities')
    st.pyplot(fig1)
    

    #Threshold condition for image centering classification
    if max_centered < 0.4:
      st.write("Prediction: This concrete has a crack")
    else:
      st.write("Prediction: This concrete has no crack")

    
    


if __name__ == "__main__":
    main()
