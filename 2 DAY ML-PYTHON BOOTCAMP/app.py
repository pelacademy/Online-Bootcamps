import os
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model

st.header('Wheat Leaf Disease Prediction Model')

disease_names = ['Black Rust',
 'Blast',
 'Brown Rust',
 'Healthy',
 'Mildew',
 'Septoria',
 'Tan spot',
 'Yellow Rust']

model = load_model("C:/leaf_classify/classifier.h5")

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + disease_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    with open(os.path.join("C:/leaf_classify/Wheat Leaf Dataset/test/", uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, width=200)
    st.markdown(classify_images(uploaded_file))
