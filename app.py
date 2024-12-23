import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

plt = platform.system()
if plt == "Linux": pathlib.WindowsPath = pathlib.PosixPath
    
#title
st.title('The model of CalassificationVehicles')

#downloadpicture
file = st.file_uploader('Download_Picture', type=['png','jpeg','gif','svg'])

if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    #prediction
    pred, pred_id, probs= model.predict(img)
    st.success(f'bashorat:{pred}')
    st.info(f'Probability: {probs[pred_id]*100:.01f}%')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
