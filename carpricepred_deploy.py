import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.write("""

# Car Price Prediction using Machine Learning
#    

""")

from PIL import Image
image = Image.open('Pricecar2.jpg')
st.image(image, use_column_width=True)

st.write("""

###  Made by ❤ Sagar Dhandare 
##    

""")



Present_Price = st.text_input("Present market price of the Car (in Lakhs)", "10")

Kms_Driven = st.text_input("Kms driven by car", "20000")

owners = st.selectbox('Number of Previous Owners?', ("0", "1", "3"))
if owners == "0":
    owners = 0
elif owners == "1":
    owners = 1
elif owners == "3":
    owners = 3

years_old = st.text_input("How Old is the Car? (in Years)", "3")

fuel_type = st.selectbox('Fuel type of Car', ("Diesel", "Petrol", "CNG"))
if fuel_type == "Diesel":
    Fuel_type_diesel = 1
    Fuel_type_Petrol = 0

elif fuel_type == "Petrol":
    Fuel_type_diesel = 0
    Fuel_type_Petrol = 1

elif fuel_type == "CNG":
    Fuel_type_diesel = 0
    Fuel_type_Petrol = 0

indivisual = st.selectbox('Are you an Indivisual or a Dealer?', ("indivisual", "Dealer"))
if indivisual =="indivisual":
    Seller_Type_Indivisual = 1
elif indivisual == "Dealer":
    Seller_Type_Indivisual = 0

Transmission = st.selectbox('What kind of transmission does it have?', ('Manual', 'Automatic'))
if Transmission =='Manual':
    Transmission_Manual = 1
else:
    Transmission_Manual = 0

if st.button("Predict"):
    pickle_in = open("Random_forest_regression_model.pkl", "rb")
    rf_classifier = pickle.load(pickle_in)

    pred123 = rf_classifier.predict([[Present_Price,Kms_Driven,owners,years_old,Fuel_type_diesel,Fuel_type_Petrol,Seller_Type_Indivisual,Transmission_Manual]])

    st.write(f"""

    ### The predicted selling price for the car is : Rs. {pred123[0]} lakhs

    """)



