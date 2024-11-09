from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
import os
import google.generativeai as genai
from PIL import Image
api_key=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

model=genai.GenerativeModel('gemini-1.5-flash')

def generate_text(input,image):
    if input!='':
        reponse=model.generate_content([input,image])
        print(reponse.text)
    else :
        reponse=model.generate_content(image)
    return reponse.text 

st.title("Text Generator using GEMINI-PRO")
input=st.text_input('Input')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
image=''
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded image",use_column_width=True)

submit=st.button('Submit')

if submit:
    response=generate_text(input,image)
    st.subheader('The response Is ')
    st.write(response)
