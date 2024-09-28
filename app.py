import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import h5py

model=tf.keras.models.load_model('model.h5')

## load the encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('OneHotEncoder_geography.pkl','rb') as file:
    OneHotEncoder_geography=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##Streamlit app
st.title("Customer Churn Prediction")

geography=st.selectbox('Geography',OneHotEncoder_geography.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,93)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
Tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[Tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],

})
geo_encoded=OneHotEncoder_geography.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=OneHotEncoder_geography.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'prediction probability:{prediction_prob:.2f}')

if prediction_prob>0.5:
    st.write('The customer is likely to churn')

else:
    st.write('The customer is not likely to churn')

