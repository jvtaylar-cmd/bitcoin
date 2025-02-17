import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('crypto_model.pkl', 'rb'))

# Create the Streamlit app
st.title('Crypto Price Prediction')

# Input field for opening price
opening_price = st.number_input('Enter Opening Price:', value=0.0)

# Prediction button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict([[np.array(opening_price)]])
    output = np.round(prediction[0], 2)

    # Display prediction
    st.success('Crypto Price should be $ {}'.format(output))
