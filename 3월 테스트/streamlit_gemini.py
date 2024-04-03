import google.generativeai as genai
import streamlit as st

GOOGLE_API_KEY="API-KEY"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


st.title('Gemini Chatbot')
input_text = st.text_input('메시지를 입력하세요.')
responses = {}

if len(input_text) == 0:
    st.text_area('챗봇 응답: ', value="메시지를 입력하세요.")

else:
    response = model.generate_content(input_text)
    responses[input_text] = response.text
    st.text_area('챗봇 응답: ', value=responses[input_text])
