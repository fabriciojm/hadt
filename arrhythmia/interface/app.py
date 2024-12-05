import streamlit as st
import pandas as pd
import requests
import numpy as np
import time

st.set_page_config(
    page_title="Arrhythmia",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto")

'''
# ❤️🫀🥼
'''

st.markdown('''
Howdy, dear heart-doctor, let's detect if your patients have arrhythmia.
''')

## CSV uploader
uploaded_file = st.file_uploader(
    "Upload your patient's ECG file below:",
    type="csv",
    help="Only 1 file at a time please, in .csv format"
    )

## Preview of uploaded CSV
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("Here's a preview of the data you just uploaded:")
    # st.write(df.T)
    # st.write(type(df.columns)) # => pandas list d'index
    # st.write(type(int(df.columns[10]))) # => pandas list d'index
    # st.write(type(df.values)) # => liste de floats
    # st.write(type(df.head(1).T[0]))
    # st.line_chart(x=df.head(1).T[0].index, y=df.head(1).T[0])
    st.line_chart(df)

    ## Button to classify heartbeats
    # st.button("Start", type="primary")
    if st.button("Start prediction"):
        # st.spinner('Loading...')
        with st.spinner('Loading...'):
            time.sleep(2)
        st.success("This is an 'S' heartbeat.")
