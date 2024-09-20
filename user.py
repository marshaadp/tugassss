import streamlit as st
import pandas as pd
from datetime import datetime

# Set title for the user interface
st.title("Input Data Reach")

# Create form for user input
with st.form("input_form"):
    # Input fields for user data
    date = st.date_input("Tanggal", datetime.today())
    reach = st.number_input("Reach", min_value=0)
    gender_l = st.number_input("Reach Gender L (Laki-laki)", min_value=0)
    gender_p = st.number_input("Reach Gender P (Perempuan)", min_value=0)
    
    # Input for different age ranges
    age_18_24 = st.number_input("Range Usia 18-24", min_value=0)
    age_25_34 = st.number_input("Range Usia 25-34", min_value=0)
    age_35_44 = st.number_input("Range Usia 35-44", min_value=0)
    age_45_54 = st.number_input("Range Usia 45-54", min_value=0)
    age_55_64 = st.number_input("Range Usia 55-64", min_value=0)
    age_65_plus = st.number_input("Range Usia 65+", min_value=0)

    # Submit button
    submitted = st.form_submit_button("Submit")

# If form is submitted, save data to a CSV file
if submitted:
    # Create a DataFrame to store the data
    new_data = pd.DataFrame({
        'Tanggal': [date],
        'Reach': [reach],
        'Gender (L)': [gender_l],
        'Gender (P)': [gender_p],
        'Range Usia 18-24': [age_18_24],
        'Range Usia 25-34': [age_25_34],
        'Range Usia 35-44': [age_35_44],
        'Range Usia 45-54': [age_45_54],
        'Range Usia 55-64': [age_55_64],
        'Range Usia 65+': [age_65_plus]
    })

    # Save the data to a CSV file (append mode)
    new_data.to_csv('user_input_data.csv', mode='a', header=False, index=False)
    
    st.success("Data berhasil disimpan!")
