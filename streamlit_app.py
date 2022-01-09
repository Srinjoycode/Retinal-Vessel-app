import streamlit as st
import numpy as np
import SessionState
from PIL import Image
import cv2
import onnxruntime


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


session = onnxruntime.InferenceSession("models/vesselNetCHASE140.onnx",
                                       providers=['CPUExecutionProvider'])
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
    results = results[0]

    results = sigmoid(results)
    results = (results > 0.5).astype('float64')

    results = results.squeeze()
    out = results

    return out


uploaded_file = st.file_uploader(label="Upload an image of the retina",
                                 type=["png", "jpeg", "jpg"])
session_state = SessionState.get(pred_button=False)
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    session_state.pred_button = True

if session_state.pred_button:
    out = make_prediction(uploaded_file)
    st.image(out, caption="The predicted vessel mask", width=None, use_column_width='always', clamp=False,
             output_format="auto")

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
