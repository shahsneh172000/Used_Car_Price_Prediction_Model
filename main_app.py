import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import base64

from yaml import load



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperaccess.com/full/33115.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
)



car_data = joblib.load('Car_data.pkl')
df = joblib.load('clean_train_data.pkl')
del df["Unnamed: 0"]
x = df.drop("Price",axis=1)
y = df['Price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=51)
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)



def car_price(year,km,seats,mileage,eng,power,location,car,fuel,trans,owner):
    index_list = []
    arr = np.zeros(98)

    arr[0] = year
    arr[1] = km
    arr[2] = seats
    arr[3] = mileage
    arr[4] = eng
    arr[5] = power

    if "Location_" + location in x.columns:
        index = np.where(x.columns == "Location_"+location)[0][0]
        arr[index] = 1
        index_list.append((index,"loc"))

    if "Car_" + car in x.columns:
        index = np.where(x.columns == "Car_" + car)[0][0]
        arr[index] = 1
        index_list.append((index,"car"))

    if "Fuel_Type_" + fuel in x.columns:
        index = np.where(x.columns == "Fuel_Type_" + fuel)[0][0]
        arr[index] = 1
        index_list.append((index,"fuel"))

    if "Transmission_" + trans in x.columns:
        index = np.where(x.columns == "Transmission_" + trans)[0][0]
        arr[index] = 1
        index_list.append((index,"trans"))

    if "Owner_Type_" + owner in x.columns:
        index = np.where(x.columns == "Owner_Type_" + owner)[0][0]
        arr[index] = 1
        index_list.append((index,"owner"))

    arr = sc.transform([arr])[0]
    return model.predict([arr])[0]



car_names = pickle.load(open('car_names.pkl','rb'))
city_names = pickle.load(open('city_names.pkl','rb'))
fuel_type = pickle.load(open('fuel_type.pkl','rb'))
owner_type = pickle.load(open('owner_type.pkl','rb'))
trans_type = pickle.load(open('trans_type.pkl','rb'))
model = joblib.load("final_rfr.pkl")




# for title
title_style = '''
<h1 style='text-align: left; color: white;font-size:70px;'>Used Car Price Prediction Model</h1>
'''
st.markdown(title_style, unsafe_allow_html=True)
# st.title("Used Car Price Prediction Model")

with st.form("my_form"):
    st.write("Please Enter Following Details To Get Estimited Car Price ")

    car = st.selectbox("Model",car_names)
    location = st.selectbox("City",city_names)
    fuel = st.selectbox("Fuel-Type",fuel_type)
    owner= st.selectbox("Owner",owner_type)
    trans= st.selectbox("Transmission",trans_type)
    year = st.text_input("Year",placeholder="Enter Year")
    km = st.text_input("Kilometer Driven",placeholder="Enter Total Kilometer Driven")
    mileage = int(car_data.loc[car_data['Car']==car]['Mileage_int'])
    seats = int(car_data.loc[car_data['Car']==car]['Seats'])
    engine = int(car_data.loc[car_data['Car']==car]['Engine_int'])
    power = int(car_data.loc[car_data['Car']==car]['Power_int'])


    submitted = st.form_submit_button("Get Price")
    if submitted:
        ans = car_price(year,km,seats,mileage,engine,power,location,car,fuel,trans,owner)
        # year,km,seats,mileage,eng,power,location,car,fuel,trans,owner
        st.write("Car Name : ",car)
        st.write("Seats : " , seats)
        st.write("Mileage : ",mileage)
        st.write("Engine(CC)" , engine , " CC")
        st.write("Power : ",power," BHP")
        st.write("Estimated Price=",round(ans,2),"Lakh")