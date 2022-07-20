import streamlit as st
import pickle
import pandas as pd

class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 

with open("pipeline.pkl", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

st.markdown("<h1 style='text-align: center; color: black;'> Credit for Loans Predictor </h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
col1, col2, col3 = st.columns([1.5, 0.5, 2.5])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    col1.write(df)
    if col2.button('Predict'):
        predict = pipeline.predict_proba(df)
        pred = pd.DataFrame({'ID': df['ID'], 'Good Loan Prob': predict[:,0], 'Bad Loan Prob': predict[:,1]})
        col3.write(pred)