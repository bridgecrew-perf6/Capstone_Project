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
    
    #Subtract pixels
    img_mean = img_normalized.mean()
    img_centered = img_normalized - img_mean
    #mean_centered = mean(img_centered)
    
    st.write(img_centered)
    plt.hist(img_centered)
    st.text('Distribution of mean centered pixel intensities')
    st.pyplot(fig)

    #Threshold condition for image centering classification
    if img_centered > 0.2:
      st.write("Prediction: This concrete has a crack")
    else:
      st.write("Prediction: This concrete has no crack")

    st.header("Conclusion")
    st.write("There were two approaches used in this model, the first was to measure the average pixel difference between the crack image and the no crack based on experimentation in jupyter notebook")
    st.write("The second approach was to subtract the pixel values from the mean and center the image. A threshold was also decided based on the centering to differentiate crack versus no crack")     
    st.write("I ran 5 images from the original dataset through the model on both techniques")
    st.write("I passed 5 images through the mean method and obtained 5 correctly classified(100%)")
    st.write("I passed the same 5 images through the pixel centering method and obtained 80% accuracy")
    st.write("However, external data unrelated to the datset gave a 60% accuracy for the average pixel method")
    st.write("And the pixel centering method was only 40% accurate")
    st.write("I discovered that the threshold does not generalize well on external data. This could be due to image resolution(quality of camera) as I observed.")
    st.write("The second approach does not generalize well to external images as well")
    st.write("The selection of the threshold will require more experimentation and additional image processing techniques.")
    st.write("I believe the change in resolution of the dataset influenced the performance of the approach on external data")
    st.header("References")
    


if __name__ == "__main__":
    main()
