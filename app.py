import streamlit as st
import pandas as pd 
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


model = load_model('saved_model.h5')

scaler = StandardScaler()

def predict_price(squarefeet,bedrooms, yearbuiltin):

    
    x = scaler.fit_transform(squarefeet)
    y = scaler.fit_transform(bedrooms)
    z = scaler.fit_transform(yearbuiltin)

    input_data = np.array([[x,y,z]])
    input_data = np.reshape(input_data, (1, 3))

    predict_price = model.predict(input_data)[0][0]
    return predict_price
    

    
st.title("Welcome to My World!")


squarefeet = st.number_input('Squarefeet',min_value=0 , step=1 )
bedrooms = st.slider('BedRooms',min_value=0, max_value=10, step=1 )
yearbuiltin = st.number_input('YearBuiltIn',min_value=0 , max_value=2023, step=1 )


if st.button('Predict Price'):
    input_1 = np.array([[squarefeet]])
    input_2 = np.array([[bedrooms]])
    input_3 = np.array([[yearbuiltin]])
    prediction = predict_price(input_1,input_2,input_3)

    in_predict = np.array([[prediction]])

    un_scaled = scaler.inverse_transform (in_predict)

    st.success(f'Predicted price is : {un_scaled:}') 