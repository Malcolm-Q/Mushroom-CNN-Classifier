from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)
from keras import layers
from tensorflow.image import resize
from PIL import Image
import streamlit as st
import urllib
import logging
from ModelLoader import load_siamese
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# tiny function to prep our inputs
def prepImage(img):
    # crop to square if needed
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = left + height
        top = 0
        bottom = height
    elif width < height:
        top = (height - width) / 2
        bottom = top + width
        left = 0
        right = width
        img = img.crop((left, top, right, bottom))

    # size to 200x200
    width, height = img.size
    if width != 200:
        img = img.resize((200,200), Image.ANTIALIAS)
    return img

# Streamlit reruns script on change, we can try run this to make it a bit smoother.
# If you wanted to use a Streamlit site as a final product I think you should use the API instead of loading the model
# But this is a standalone website deployment and you could easily swap it out for an API request if needed
try:
    if siamese_model is None:
        siamese_model, class_to_species = load_siamese()
except:
    pass


st.title('Mushroom Classifier')
st.write('Upload two images of a mushroom from different perspectives, \n\n ideally one looking down on the mushroom and where you can see the gills. \n\n Ensure the mushroom is centered in the photo and please take photos of mushrooms in their environments, the neural network takes the background/substrate into account.')

# user uploads images
with st.form("my_form"):
    st.write("Inside the form")
    uploaded_file = st.file_uploader('Image of the top perspective', type=['jpg', 'jpeg', 'png'])
    uploaded_file2 = st.file_uploader('Image of the bottom perspective', type=['jpg', 'jpeg', 'png'])
    submitted = st.form_submit_button("Predict")

# When both files are uploaded we show them and make our prediction
if uploaded_file is not None and uploaded_file2 is not None:
    image = Image.open(uploaded_file)
    image2 = Image.open(uploaded_file2)
    image = prepImage(image)
    image2 = prepImage(image2)


    st.image(image, caption='First Image', use_column_width=True)
    st.image(image2, caption='Second Image', use_column_width=True)

    # input is [(None,200,200,3),(None,200,200,3)]
    prediction = siamese_model.predict([np.expand_dims(np.array(image),axis=0), np.expand_dims(np.array(image2),axis=0)]).tolist()
    st.write(f'I am {round(prediction[0][np.argmax(prediction[0])]*100,1)}% confident that the mushroom you pictured is {class_to_species[str(np.argmax(prediction[0]))]}')
    st.write('Please verify your findings with local foragers in your community and never eat a mushroom YOU are not 100% certain of')

    # embed class to url and search for more info.
    query = urllib.parse.quote(class_to_species[str(np.argmax(prediction))])
    url = f'[See more images of {class_to_species[str(np.argmax(prediction))]}](https://www.google.com/search?q={query})'
    st.markdown(url,unsafe_allow_html=True)