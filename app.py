"""
# An app to visualize and run a vessel detection model
Author : Srinjoy Bhuiya
"""
import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import SessionState
import io
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# TODO remove this code for azure auth onnx model
import onnxruntime

session = onnxruntime.InferenceSession("VesselNet_CHASE_800_65epochs.onnx",
                                       providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                  'CPUExecutionProvider'])

st.title('Retinal Vessel Segmentation')
st.header("Identify the vessels in a retinal fundus image photos!")


@st.cache  # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(uploaded_file):
    """
    Takes an uploaded file and uses model (a trained model) to make a
    prediction.
    Returns:
     image (preproccessed)
    """
    img = np.array(Image.open(uploaded_file))
    img = np.array(cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_CUBIC))
    img = np.float32(img / 255.0)

    if img.ndim == 3:
        img = np.expand_dims(img, 0)

    img = img.transpose(0, 3, 1, 2)

    results = session.run([], {"input": img})
    results = torch.from_numpy(results[0])

    results = torch.sigmoid(results)
    results = (results > 0.5).float()

    results = results.squeeze()
    out = results.detach().cpu().numpy()

    # download_image=np.ndarray.tobytes((out))
    # btn = st.download_button(
    #     label="Download the vascular structure",
    #     data=download_image,
    #     file_name="vessel_mask.png",
    #     mime="image/png"
    # )

    return out


# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of the retina",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True

# And if they did...
if session_state.pred_button:
    out = make_prediction(uploaded_file)
    st.image(out, caption="The predicted vessel mask", width=None, use_column_width='always', clamp=False,
             output_format="auto")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")


    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What is the particular issue?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)



