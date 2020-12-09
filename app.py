import pandas as pd
import numpy as np
import streamlit as st
import pickle
st.write("""
# Diabetes Prediction using Machine Learning
#    
""")

from PIL import Image
image = Image.open('diabetes.jpg')
st.image(image, use_column_width=True)

st.write("""
###  Made by ❤ Sagar Dhandare 
##    
""")



Pregnancies = st.text_input("Pregnancies", "10")

Glucose = st.text_input("Glucose", "20000")

BloodPressure = st.text_input("BloodPressure", "20000")

SkinThickness = st.text_input("SkinThickness", "20000")

Insulin = st.text_input("Insulin", "20000")

BMI = st.text_input("BMI", "20000")

DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction", "20000")

Age = st.text_input("Age", "20000")



if st.button("Predict"):
    pickle_in = open("diabetes.pkl", "rb")
    rf_classifier = pickle.load(pickle_in)

    pred123 = rf_classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    if pred123 == 1:
        st.write(f"""
        ### Sorry to say that yor are tested POSITIVE for the DIBETES. Don't worry consult a doctor soon!! You will be alright..
        
        """)
    elif pred123 == 0:
        st.write(f"""
        ### Congrutulations you are tested NEGATIVE.
        """)
    else:
        st.write(f"""
        ### Something is wrong Sagar
        """)

