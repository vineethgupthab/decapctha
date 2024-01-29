import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers


# Streamlit
st.set_page_config(page_title="Decpatcha")
st.header("De-captcha Application")
uploaded_image = st.file_uploader('Upload the Image', type=['png','jpg'], accept_multiple_files=False)
if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded captcha')
submit=st.button("De-captcha the Image")

# TF model
prediction_model = tf.keras.models.load_model('decaptcha_model')

img_width = 200
img_height = 50
def encode_sample(img_path): 
	# Read the image 
	img = tf.io.read_file(img_path) 
	# Converting the image to grayscale 
	img = tf.io.decode_png(img, channels=1) 
	img = tf.image.convert_image_dtype(img, tf.float32) 
	# Resizing to the desired size 
	img = tf.image.resize(img, [img_height, img_width]) 
	# Transposing the image 
	img = tf.transpose(img, perm=[1, 0, 2])
	img = (img * 255).numpy().astype("uint8")
	return img

max_length = 5
def decode_batch_predictions(pred): 
	input_len = np.ones(pred.shape[0]) * pred.shape[1] 
	results = keras.backend.ctc_decode(pred, 
									input_length=input_len, 
									greedy=True)[0][0][ 
		:, :max_length 
	] 
	output_text = [] 
	for res in results: 
		res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8") 
		output_text.append(res) 
	return output_text 


with open('char_img.pkl', 'rb') as f:
    char_img = pickle.load(f)
char_to_num = layers.StringLookup( 
	vocabulary=list(char_img), mask_token=None
) 

num_to_char = layers.StringLookup( 
	vocabulary=char_to_num.get_vocabulary(), 
	mask_token=None, invert=True
) 


if submit:
	if uploaded_image is not None:
		test_img = encode_sample(uploaded_image)
		test_img = np.reshape(test_img, (-1, 200, 50, 1))
		result = decode_batch_predictions(prediction_model.predict(test_img))
		st.write(result)
