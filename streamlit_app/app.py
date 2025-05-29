import streamlit as st 
import pandas as pd 
import requests 

def model_predict(file_uploader_obj):
    with st.spinner("Running the prediction model..."):
        try:
             files = {
                 'file': (
                        file_uploader_obj.name,  
                        file_uploader_obj.getvalue(),  
                        'text/csv'  
                )
             }
             response = requests.post("http://api:5000/predict", files=files)
             response.raise_for_status()
             pred_result = response.json()
             st.write(pred_result)
        except Exception as e:
             st.error(f"An error occurred: {e}")

st.header("Credit Fraud Detection", divider=True)

st.write("Upload CSV file with transactions to detect fraud")


uploaded_csv = st.file_uploader("Upload CSV", type="csv")

if uploaded_csv is not None: 
    st.button("Predict", use_container_width=True, on_click=lambda: model_predict(uploaded_csv))


