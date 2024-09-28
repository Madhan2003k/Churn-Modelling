import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import h5py

model=tf.keras.models.load_model('regmodel.h5')

## load the encoder and scaler
with open('Onehotencoder_geo.pkl','rb')as file:
   Onehotencoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb')as file:
   label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


##streamlit app
st.title('Estimated salary prediction')
geography=st.selectbox('Geography',Onehotencoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,93)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
Tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
exited=st.selectbox('Exited',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[Tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
    })

geo_encoded=Onehotencoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=Onehotencoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


input_data_scaled=scaler.transform(input_data)
prediction=model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write(f'Predicted Estimated salary: ${predicted_salary:.2f}')