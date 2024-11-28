import streamlit as st
import pandas as pd
import requests
import numpy as np

'''
# 🫀🥼
'''

st.markdown('''
Howdy, dear heart-doctor, let's detect if your patients have arrhythmia.
''')


uploaded_file = st.file_uploader(
    "Upload your patient's ECG file below:",
    type="csv",
    help="Only 1 file at a time please, in .csv format"
    )
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    df = dataframe.drop(columns="target")
    st.markdown("Here's a preview of the data you just uploaded:")
    st.write(df.head(1).values)
    # st.write(type(df.columns)) # => pandas list d'index
    # st.write(type(int(df.columns[10]))) # => pandas list d'index
    # st.write(type(df.values)) # => liste de floats
    # st.write(type(df.head(1).T[0]))
    # st.line_chart(x=df.head(1).T[0].index, y=df.head(1).T[0])
    data = df.head(1).T
    st.line_chart(data[0])
