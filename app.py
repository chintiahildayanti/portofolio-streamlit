import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Membuat menu navigasi
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Page 2"],  # required
        icons=["house", "file"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# Halaman Home
if selected == "Home":
    st.title("Home Page")
    st.write("This is the main page, which contains visualizations of adult census income of United States Citizens")

    # Title
    st.title("Census Income Of United States Citizens")

    @st.cache_resource
    # Load data (pastikan untuk mengganti 'your_data.csv' dengan path yang benar ke file dataset Anda)
    def load_data():
        df = pd.read_csv('adult_census.csv')  # Ganti dengan dataset Anda
        return df

    df = load_data()

# Show data preview
# Input interaktif untuk rentang usia
    age_range = st.slider('Select Age Range', min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(int(df['age'].min()), int(df['age'].max())))

# Filter DataFrame berdasarkan rentang usia yang dipilih
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]

# Age distribution plot
    st.subheader("Age Distribution")
    plt.figure(figsize=(10, 6))
    filtered_df['age'].plot(kind='hist', bins=30, title='Age Distribution', color='lightblue')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Sidebar untuk input interaktif dengan selectbox
    st.sidebar.header('Filter Options')
    education_options = df['education'].unique()  # Ambil daftar tingkat pendidikan
    selected_education = st.sidebar.selectbox('Select Education Level:', education_options)

# Filter data berdasarkan pilihan pengguna
    filtered_data = df[df['education'] == selected_education]

# Visualisasi distribusi income berdasarkan tingkat pendidikan yang dipilih
    st.subheader(f"Income Distribution for Education Level: {selected_education}")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='income', data=filtered_data, palette=['lightblue', 'blue'])
    plt.title(f'Income Distribution for {selected_education}')
    st.pyplot(plt)


# Halaman Page 2
if selected == "Page 2":
    st.title("Page 2")
    st.write("This is the second page, containing the application for predicting social assistance recipients")

    @st.cache_resource
# Load model
    def load_model():
        xgb_model = joblib.load("xgboost_model.pkl")
        return xgb_model

    model = load_model()
# Title
    st.title("United States Government Social Assistance App")

# Input features
    st.write("## Input Your Data Here")
    age = st.number_input('Age', min_value=17, max_value=90, value=30)

# Workclass mapping and selection
    marital_status_mapping = {
        21959: 'Married-civ-spouse',
        15536: 'Never-married',
        6523: 'Divorced',
        1497: 'Separated',
        1443: 'Widowed',
        600: 'Married-spouse-absent',
        34: 'Married-AF-spouse'
    }
    selected_marital_status = st.selectbox('Marital Status', options=list(marital_status_mapping.values()))
    marital_status_encoded = list(marital_status_mapping.keys())[list(marital_status_mapping.values()).index(selected_marital_status)]


    education_mapping ={
        1:'Preschool',
        2:'1st-4th',
        3:'5th-6th',
        4:'7th-8th',
        5:'9th', 
        6:'10th',
        7:'11th',
        8:'12th',
        9:'HS-grad',
        10:'Some-college',
        11:'Assoc-voc',
        12:'Assoc-acdm', 
        13:'Bachelors',
        14:'Masters',
        15:'Prof-school',
        16:'Doctorate'
    }
    selected_education = st.selectbox('Education', options=list(education_mapping.values()))
    education_encoded = list(education_mapping.keys())[list(education_mapping.values()).index(selected_education)]


    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=0)

# Input data (ensure all necessary features are provided for the model)
# Dummy values (0) can be replaced with additional user inputs as required by the model
    input_data = np.array([[age, 0, education_encoded, capital_gain, 0, 0, 0, 0, 0, marital_status_encoded, 0, 0, 0]])

# Prediction button
    if st.button('Predict'):
        prediction = model.predict(input_data)
    # Show prediction result
        if prediction[0] == 1:
            st.success("You are entitled to social assistance")
        else:
            st.error("You are not entitled to social assistance")
