from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
import os
import google.generativeai as genai
api_key=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

model=genai.GenerativeModel('gemini-pro')

def generate_text(prompt):
    reponse=model.generate_content(prompt)
    print(reponse.text)
    return reponse.text 

st.title("Text Generator using GEMINI-PRO")
input=st.text_input('Input')
submit=st.button('Submit')
if submit:
    response=generate_text(input)
    st.subheader('The response Is ')
    st.write(response)