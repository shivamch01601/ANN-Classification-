# Import libraries
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    encoder_geo = pickle.load(file)

## Load Scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 

#--------------
# Streamlit app  
#--------------

# Title
st.title("Customer Churn Prediction Model")

# Inputs
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# One hot encoder for geography
geo_encoded = encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns = encoder_geo.get_feature_names_out(['Geography'])
)

# Combine it in input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Arrange columns in the same order used during training
input_data = input_data[scaler.feature_names_in_]

# Scale data
scaled_data = scaler.transform(input_data)

# predication
pred = model.predict(scaled_data, verbose=0)
pred_prob = pred[0][0]

st.write(f'Churn Probability : {pred_prob:.2f}')

if pred_prob > 0.5:
    st.write('The Customer is likely to churn')
else:
    st.write('The Customer is not likely to churn')