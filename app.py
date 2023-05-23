
import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Run Command  -> streamlit run app.py
df = pd.read_csv('encoded_car_df.csv')
load_model = pickle.load(open('rf_model.pkl','rb'))
st.title('Car Price Predictor')
st.header('Fill the details for the price')
#Car_Model = st.selectbox('Car Model',df['brand'].unique())
Year = st.selectbox('Year',df['year'].unique())
Km_driven = st.selectbox('Total Driven',df['km_driven'].unique())
fuel = st.selectbox('Fuel Type',df['fuel_encoded'].unique())
seller_type = st.selectbox('Seller Type',df['seller_type_encoded'].unique())
transmission = st.selectbox('Transmission',df['transmission_encoded'].unique())
owner = st.selectbox('Owner Type',df['owner_encoded'].unique())

if st.button("Car Price"):
    test_data = np.array([Year,Km_driven,fuel,seller_type,transmission,owner])
    test_data = test_data.reshape([1,6])
    st.success(load_model.predict(test_data)[0])
