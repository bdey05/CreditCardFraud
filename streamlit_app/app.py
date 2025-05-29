import streamlit as st 
import pandas as pd 
import requests 

st.header("Credit Fraud Detection", divider=True)

st.write("Upload CSV file with transactions to detect fraud")


uploaded_csv = st.file_uploader("Upload CSV", type="csv")

st.button("Predict", use_container_width=True)

