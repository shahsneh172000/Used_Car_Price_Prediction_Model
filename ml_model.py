import joblib 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

model = joblib.load("usedcar_ml_model2.pkl")
df = pd.read_csv("clean_train_data.csv")
del df['Unnamed: 0']
x = df.drop("Price",axis=1)
y = df['Price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=51)

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

def car_price(year,km,seats,mileage,eng,power,location,brand,mod,fuel,trans,owner):
    index_list = []
    arr = np.zeros(113)

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

    if "Brand_" + brand in x.columns:
        index = np.where(x.columns == "Brand_" + brand)[0][0]
        arr[index] = 1
        index_list.append((index,"brand"))

    if "Model_" + mod in x.columns:
        index = np.where(x.columns == "Model_" + mod)[0][0]
        arr[index] = 1
        index_list.append((index,"mod"))

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

# print(car_price(2015,41000,5,19.67,1586,126.2,"Pune","Hyundai","Creta","Diesel","Automatic","First"))