# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: siddhardhan
"""

import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import json
import streamlit as st
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import hashlib


# login and registration


# Define the path for the JSON file to store user data
USER_DATA_FILE = 'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/login_rgister_data/users.json'

# Function to load user data from JSON file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

# Function to save user data to JSON file
def save_user_data(user_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_data, f)

# Hash password using SHA-256
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Login functionality
def login(username, password):
    user_data = load_user_data()
    if username in user_data and user_data[username] == hash_password(password):
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        return True
    return False

# Registration functionality
def register(username, password):
    user_data = load_user_data()
    if username in user_data:
        return False  # Username already exists
    user_data[username] = hash_password(password)
    save_user_data(user_data)
    return True

# Streamlit UI for Login and Registration
def login_register_ui():
    st.title("Login and Registration System")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Selection between Login and Register
    choice = st.sidebar.selectbox("Select Action", ["Login", "Register"])

    if choice == "Register":
        st.subheader("Create a New Account")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")

        if st.button("Register"):
            if register(new_username, new_password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Try a different one.")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login(username, password):
                st.success(f"Welcome, {username}!")
            else:
                st.error("Incorrect username or password.")





# Load and preprocess the dataset
dataset = pd.read_csv("C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/datasets/general.csv")
X = dataset.drop('Disease', axis=1)
y = dataset['Disease']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


# loading the saved models

diabetes_model = pickle.load(open(f'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/models/heart_disease_model1.sav','rb'))

parkinsons_model = pickle.load(open(f'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/models/parkinsons_model.sav', 'rb'))

def predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion):
    # Prepare user input as a single-row DataFrame
    user_input = pd.DataFrame({
        'Temp': [temp_f],
        'Pulserate': [pulse_rate_bpm],
        'Vomiting': [vomiting],
        'YellowishUrine': [yellowish_urine],
        'Indigestion': [indigestion]
    })

    # Standardize the user input
    user_input = scaler.transform(user_input)

    # Make prediction
    predicted_disease = model.predict(user_input)[0]
    disease_names = { 0: 'Heart Disease',1: 'Viral Fever/Cold', 2: 'Jaundice', 3: 'Food Poisoning',4: 'Normal'}
    return disease_names[predicted_disease]

def show_attribute_descriptions():
    attribute_descriptions = {
        "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
        "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
        "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
        "MDVP:Jitter(%)": "Several measures of variation in fundamental frequency",
        "MDVP:Jitter(Abs)": "Several measures of variation in fundamental frequency",
        "MDVP:RAP": "Several measures of variation in fundamental frequency",
        "MDVP:PPQ": "Several measures of variation in fundamental frequency",
        "Jitter:DDP": "Several measures of variation in fundamental frequency",
        "MDVP:Shimmer": "Several measures of variation in amplitude",
        "MDVP:Shimmer(dB)": "Several measures of variation in amplitude",
        "Shimmer:APQ3": "Several measures of variation in amplitude",
        "Shimmer:APQ5": "Several measures of variation in amplitude",
        "MDVP:APQ": "Several measures of variation in amplitude",
        "Shimmer:DDA": "Several measures of variation in amplitude",
        "NHR": "Two measures of ratio of noise to tonal components in the voice",
        "HNR": "Two measures of ratio of noise to tonal components in the voice",
        "status": "Health status of the subject (one) - Parkinson's, (zero) - healthy",
        "RPDE": "Two nonlinear dynamical complexity measures",
        "D2": "Two nonlinear dynamical complexity measures",
        "DFA": "Signal fractal scaling exponent",
        "spread1": "Three nonlinear measures of fundamental frequency variation",
        "spread2": "Three nonlinear measures of fundamental frequency variation",
        "PPE": "Three nonlinear measures of fundamental frequency variation",
    }

    st.header("Attribute Descriptions")
    for attribute, description in attribute_descriptions.items():
        st.write(f"**{attribute}**: {description}")


def calculate_bmi(weight, height):
    bmi = weight / (height / 100) ** 2
    return bmi

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# sidebar for navigation
def main():
    with st.sidebar:
        image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/navbar.png')
        st.image(image,width =180)
        # Streamlit UI for Login and Registration
                        
        selected = option_menu('Disease Diagnosis and Recommendation System',
                              
                              [
                               'GENERAL',
                               'Diabetes Prediction',
                               'Heart Disease Prediction',
                               'Parkinsons Prediction',
                               'BMI CALCULATOR',
                               'Disease prediction and Doctor recommendation',
                               'Appointment Booking'
                               ,'Weakly Report'],
                              icons=['dashboard','activity','heart','person','line-chart'],
                              default_index=0)
        

    if(selected == 'GENERAL'):
        st.title("General Diagnosis") 
        st.write("Please enter the following information:")
        col1,col2 = st.columns([2,1])
        with col1:
         # Get user input
           temp_f = st.text_input("Temperature (F):", value=0)
           pulse_rate_bpm = st.text_input("Pulse rate (bpm):", value=0)
           st.write("Check the Symptoms you have below:")
           vomiting = st.checkbox("Vomiting")
           yellowish_urine = st.checkbox("Yellowish Urine")
           indigestion = st.checkbox("Indigestion")
   
       # Predict disease based on user input
           if st.button("Test Result"):
               predicted_disease = predict_disease(temp_f, pulse_rate_bpm, vomiting, yellowish_urine, indigestion)
               medicine_recommendations = {
       'Heart Disease': 'Follow a heart-healthy diet and exercise regularly.\nIt is crucial to attend all scheduled appointments with your doctor for proper monitoring and management.',
       'Viral Fever\Cold': 'Get plenty of rest and stay hydrated.\nIf your fever persists or is accompanied by severe symptoms, visit a doctor for proper evaluation and treatment.',
       'Jaundice': 'Rest, stay well-hydrated, and follow a balanced diet.\n If you notice yellowing of the skin or eyes (jaundice), seek medical attention immediately for proper diagnosis and treatment.',
       'Food Poisoning': 'Stay hydrated and avoid solid foods until symptoms subside.\nIf you experience severe symptoms, seek medical attention promptly for proper evaluation and treatment.',
       'Normal': 'Maintain a healthy lifestyle with regular exercise and a balanced diet.\nEven if you are feeling well, have regular check-ups with your doctor to monitor your overall health.'
        } 
           
   
       
       # Show the pop-up box with disease prediction and medicine recommendation
               if predicted_disease in medicine_recommendations:
                   medicine_recommendation = medicine_recommendations[predicted_disease]
                   st.info(f"Predicted Disease: {predicted_disease}")
                   with st.expander("Medicine Recommendation:"):
                       st.info(f"Medicine Recommendation: {medicine_recommendation}")
               else:
                   st.warning("Unknown disease prediction. Please check your input and try again.")
               #st.write(f"Predicted Disease: {predicted_disease}")    
        
        with col2:
            image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/general.png')
            st.image(image,width =500)
        
    # Diabetes Prediction Page
    if (selected == 'Diabetes Prediction'):
        
        # page title
        st.title('Diabetes Prediction')
        
        
        # getting the input data from the user
        col1, col2, col3,col4= st.columns(4)
        
        with col1:
            Pregnancies = st.text_input('No of Pregnancies')
            
        with col2:
            Glucose = st.text_input('Glucose Level')
        
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        
        with col2:
            Insulin = st.text_input('Insulin Level')
        
        with col3:
            BMI = st.text_input('BMI')
        
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
        with col2:
            Age = st.text_input('Age')
        
        with col4:
            image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/diabetes.png')
            st.image(image,width = 400)  

        # code for Prediction
        diab_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
              st.success('The person is diabetic')
              with st.expander("Medicine Recommendation:"):
                    st.info(f"Medicine Recommendation: {'Please Consult a Medical Profesional. Please follow a balanced and healthy diet. It is important to exercise regularly.'}")
            else:
              st.success('The person is not diabetic')
        
           
        #st.success(diab_diagnosis)
    
    
    
    
    # Heart Disease Prediction Page
    if (selected == 'Heart Disease Prediction'):
        
        # page title
        st.title('Heart Disease Prediction')
        
        col1, col2, col3 ,col4= st.columns(4)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            sex_options = ['Male','Female']
            sex = st.selectbox('Sex',sex_options)
            
        with col3:
            cp = st.text_input('Chest Pain type(1,2,3,4)')
            
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
            
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
            
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
            
        with col3:
            exang = st.text_input('Exercise Induced Angina')
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
            
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
            
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
            
        with col1:
            thal = st.text_input('Results of Nuclear Stress Test(0,1,2)')
            
        with col4:
            image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/heart.png')
            st.image(image,width =350)

         
         
        # code for Prediction
        heart_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Heart Disease Test Result'):
            sex_mapping = {'Male': 1, 'Female': 0}
            sex_numeric = sex_mapping[sex]
            input_data = [float(age), float(sex_numeric), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                  float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            input_data = np.array(input_data).reshape(1, -1)
            heart_prediction = heart_disease_model.predict(input_data)
            if heart_prediction[0] == 1:
                st.success('The person is having heart disease')
                with st.expander("Medicine Recommendation:"):
                    st.info(f"Medicine Recommendation: {'SEEK IMMEDIATE MEDICAL ATTENTION. We encourage you to make positive lifestyle changes to reduce risk factors for Heart Disease.'}")
            else:
                st.success('The person does not have any heart disease')
            
        #st.success(heart_diagnosis)
            
         
        
    
    # Parkinson's Prediction Page
    if (selected == "Parkinsons Prediction"):
        
        # page title
        st.title("Parkinson's Disease Prediction")
        st.subheader('Enter the details of your Biomedical Voice Measurement Test:')
        col1, col2, col3, col4, col5,col6 = st.columns(6)  
        
        with col1:
            fo = st.text_input('Fo(Hz)')
            
        with col2:
            fhi = st.text_input('Fhi(Hz)')
            
        with col3:
            flo = st.text_input('Flo(Hz)')
            
        with col4:
            Jitter_percent = st.text_input('Jitter(%)')
            
        with col5:
            Jitter_Abs = st.text_input('Jitter(Abs)')
            
        with col1:
            RAP = st.text_input('RAP')
            
        with col2:
            PPQ = st.text_input('PPQ')
            
        with col3:
            DDP = st.text_input('DDP')
            
        with col4:
            Shimmer = st.text_input('Shimmer')
            
        with col5:
            Shimmer_dB = st.text_input('Shimmer(dB)')
            
        with col1:
            APQ3 = st.text_input('APQ3')
            
        with col2:
            APQ5 = st.text_input('APQ5')
            
        with col3:
            APQ = st.text_input('APQ')
            
        with col4:
            DDA = st.text_input('DDA')
            
        with col5:
            NHR = st.text_input('NHR')
            
        with col1:
            HNR = st.text_input('HNR')
            
        with col2:
            RPDE = st.text_input('RPDE')
            
        with col3:
            DFA = st.text_input('DFA')
            
        with col4:
            spread1 = st.text_input('spread1')
            
        with col5:
            spread2 = st.text_input('spread2')
            
        with col1:
            D2 = st.text_input('D2')
            
        with col2:
            PPE = st.text_input('PPE')
            
        with col6:
            image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/parkinsons.png')
            st.image(image,width =350)

        
        # code for Prediction
        parkinsons_diagnosis = ''
        
        # creating a button for Prediction    
        if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
              st.success('The person has Parkinsons disease')
              with st.expander("Medicine Recommendation:"):
                st.info(f"Medicine Recommendation: {'CONSULT A NEUROLOGIST. Please take the prescribed medications. We also recommend physical and occupational therapy to improve mobility.'}")
            else:
              parkinsons_diagnosis = "The person does not have Parkinson's disease"
        if st.button("Show Attribute Descriptions"):
            show_attribute_descriptions()    

             
        #st.success(parkinsons_diagnosis)
        
    # selected bmi calaculator
    
        
    if (selected == 'BMI CALCULATOR'):
        
        st.title("BMI CALCULATOR")

        st.write("Body Mass Index (BMI) is a measure of body fat based on height and weight.")
        st.write("Use this calculator to find out your BMI category.")
        col1,col2 = st.columns([2,1])
        with col1:
            weight = st.text_input("Enter your weight (in kilograms)")
            height = st.text_input("Enter your height (in centimeters)")
        
            if st.button("Calculate BMI"):
                
                weight = float(weight)
                height = float(height)
                bmi = calculate_bmi(weight, height)
                category = interpret_bmi(bmi)
        
                st.write("### Results")
                st.write(f"Your BMI: {bmi:.2f}")
                st.write(f"Category: {category}")
        with col2:
            image = Image.open('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/images/bmi.png')
            st.image(image,width =350)   

    
    # if selected doctor allocation system
    
    if(selected=='Disease prediction and Doctor recommendation'):
    
# Load datasets
        try:
            training_dataset = pd.read_csv('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/my_project/Training.csv')
            test_dataset = pd.read_csv('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/my_project/Testing.csv')
            doc_dataset = pd.read_csv('C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/my_project/doctors_dataset.csv', names=['Name', 'Description'])
        except FileNotFoundError:
            st.error("One or more required files are missing. Please make sure 'Training.csv', 'Testing.csv', and 'doctors_dataset.csv' are available.")
            st.stop()

# Separate features and target labels
        X = training_dataset.iloc[:, 0:132].values
        y = training_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
        dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()

# Encoding string values to integer constants
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

# Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Initialize and train the Decision Tree Classifier
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

# Get the feature columns
        cols = training_dataset.columns[:-1]

# Set up doctor information
        diseases = pd.DataFrame(dimensionality_reduction.index, columns=['prognosis'])
        doctors = pd.DataFrame()
        doctors['name'] = doc_dataset['Name']
        doctors['link'] = doc_dataset['Description']
        doctors['disease'] = diseases['prognosis']

# Define Streamlit UI components
        st.title("Healthcare Diagnostic Bot")

        def execute_bot():
            st.write("Please respond with 'yes' or 'no' for the following symptoms:")

        def print_disease(node):
            node = node[0]
            val = node.nonzero() 
            disease = labelencoder.inverse_transform(val[0])
            return disease

        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
            ]
            symptoms_present = []

            def recurse(node, depth):
                if tree_.feature[node] != -2:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                
                    ans = st.radio(f"Do you have: {name}?", options=('Select', 'yes', 'no'), key=f"symptom_{node}")
                    # Ensure that the user selects an option
                    if ans == 'Select':
                        st.warning("Please select 'yes' or 'no' to proceed.")
                        return  # Stop further recursion until an answer is selected

                    val = 1 if ans == 'yes' else 0
                
                    if val <= threshold:
                        recurse(tree_.children_left[node], depth + 1)
                    else:
                        symptoms_present.append(name)
                        recurse(tree_.children_right[node], depth + 1)
                else:
                    present_disease = print_disease(tree_.value[node])
                    st.write(f"You may have: {present_disease[0]}")
                    st.write("""
                    ### Precautions:
                    - **Balanced Diet:** Eat a nutritious diet rich in fruits, vegetables, lean proteins, and whole grains.
                    - **Stay Hydrated:** Drink sufficient water daily.
                    - **Regular Exercise:** Engage in moderate physical activity.
                    - **Adequate Sleep:** Get at least 7-8 hours of sleep.
                    - **Consult a Doctor:** Seek medical advice for any health concerns.
                    - **Complete Prescribed Treatments:** Always complete the full course of any prescribed medication.
                    """)
                
                    red_cols = dimensionality_reduction.columns 
                    symptoms_given = red_cols[dimensionality_reduction.loc[present_disease].values[0].nonzero()]
                
                    st.write("### Symptoms Present:")
                    st.write(list(symptoms_present))
                
                    st.write("### Symptoms Expected:")
                    st.write(list(symptoms_given))
                
                    confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
                    st.write(f"**Confidence Level:** {confidence_level * 100:.2f}%")

                    st.write("#### Recommended Doctor:")
                    row = doctors[doctors['disease'] == present_disease[0]]
                    st.write(f"Consult: {row['name'].values[0]}")
                    st.write(f"Visit: {row['link'].values[0]}")
        
            recurse(0, 1)
    
        tree_to_code(classifier, cols)

# Replace button with a checkbox for persistent state
        if st.checkbox("Start Diagnosis"):
            execute_bot()
                

# Appointment booking

    if(selected=='Appointment Booking'):
        APPOINTMENT_FILE = 'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/appoinmentsappointments.json'

        EMAIL_ADDRESS = st.text_input("Enter your Email-id:")
        EMAIL_PASSWORD = st.text_input("Enter password")


        def load_appointments():
            try:
                with open(APPOINTMENT_FILE, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return []


        def save_appointments(appointments):
   
            with open(APPOINTMENT_FILE, 'w') as f:
                json.dump(appointments, f, indent=4)

        def send_email(to_email, subject, message):
    
            try:
                msg = MIMEMultipart()
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = to_email
                msg['Subject'] = subject
                msg.attach(MIMEText(message, 'plain'))
        
        # Establish connection to the server
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
                server.quit()
                st.success("Confirmation email sent to doctor!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

        def add_appointment(doctor, doctor_email, patient, date, time):
   
            appointments = load_appointments()

    # Check for conflicts
            for appointment in appointments:
                if (
                        appointment["doctor"] == doctor and
                        appointment["date"] == date and
                        appointment["time"] == time
                        ):
                    st.warning("This appointment time is already booked with the selected doctor. "
                       "Please choose a different time or date.")
                    return  # Exit the function without saving

    # If no conflict, add the appointment
            new_appointment = {
            "doctor": doctor,
            "doctor_email": doctor_email,
            "patient": patient,
            "date": date,
            "time": time
            }
            appointments.append(new_appointment)
            save_appointments(appointments)
            st.success("Appointment added successfully!")

    # Send confirmation email to the doctor
            email_subject = "New Appointment Confirmation"
            email_message = (f"Dear Dr. {doctor},\n\n"
                     f"You have a new appointment.\n"
                     f"Patient: {patient}\n"
                     f"Date: {date}\n"
                     f"Time: {time}\n\n"
                     "Best regards,\nAppointment Booking System")
            send_email(doctor_email, email_subject, email_message)
    


        def delete_appointment(index):
    
                appointments = load_appointments()
                if 0 <= index < len(appointments):
                    deleted_appointment = appointments.pop(index)
                    save_appointments(appointments)
                    st.success(f"Deleted appointment for {deleted_appointment['patient']} with Dr. {deleted_appointment['doctor']}")
                else:
                    st.error("Invalid index. Please try again.")


# Streamlit app
        st.title("Doctor's Appointment Booking System")

# Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Add Appointment", "Show Appointments", "Delete Appointment"])

# Add Appointment tab
        with tab1:
            st.header("Add Appointment")
            doctor = st.text_input("Doctor's Name")
            doctor_email = st.text_input("Doctor's Email")
            patient = st.text_input("Patient's Name")
            date = st.date_input("Appointment Date")
            time = st.time_input("Appointment Time")

            if st.button("Add Appointment"):
                if doctor and patient:
            # Format date and time as strings
                    date_str = date.strftime("%Y-%m-%d")
                    time_str = time.strftime("%H:%M")
                    add_appointment(doctor,doctor_email, patient, date_str, time_str)
            else:
                st.error("Please enter all required details.")

# Show Appointments tab
        with tab2:
            st.header("All Appointments")
            appointments = load_appointments()
            if appointments:
                for idx, appointment in enumerate(appointments):
                    st.write(f"{idx + 1}. Doctor: {appointment['doctor']}, Patient: {appointment['patient']}, "
                     f"Date: {appointment['date']}, Time: {appointment['time']}")
            else:
                st.info("No appointments found.")

# Delete Appointment tab
        with tab3:
            st.header("Delete Appointment")
            appointments = load_appointments()
            if appointments:
                selected_appointment = st.selectbox(
            "Select an appointment to delete",
            [f"{i + 1}. Doctor: {app['doctor']}, Patient: {app['patient']}, "
             f"Date: {app['date']}, Time: {app['time']}" 
             for i, app in enumerate(appointments)]
            )
        
            index_to_delete = int(selected_appointment.split(".")[0]) - 1
            if st.button("Delete Appointment"):
                delete_appointment(index_to_delete)
            else:
                st.info("No appointments to delete.")
                
    # weakly report system

    if(selected=='Weakly Report'):
        # Load or create a data file for weekly records
        
        DATA_FILE = 'C:/Users/soham/OneDrive/Desktop/mutiple_disease_system/Disease-Diagnosis-and-Recommendation-System-main/datasets/health.csv'

# Function to send email
        def send_email(subject, message, recipient):
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login("sohamdeshmukh8391@gmail.com", "ewsw kueg hbya zvew")  # Use the App Password here
                server.sendmail("sohamdeshmukh8391@gmail.com", recipient, f'Subject: {subject}\n\n{message}')
                server.quit()
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Error sending email: {e}")


# Initialize or load health data
        if os.path.exists(DATA_FILE):
            health_data = pd.read_csv(DATA_FILE)
        else:
            health_data = pd.DataFrame(columns=['Date', 'Symptoms', 'Weight', 'Blood Pressure', 'Sleep', 'Exercise'])

        st.title("Weekly Personal Health Tracker")

# User inputs for current week
        with st.form("Health Form"):
            date = datetime.now().strftime("%Y-%m-%d")
            symptoms = st.text_input("Enter any symptoms you have:")
            weight = st.number_input("Enter your weight (kg):", min_value=0.0)
            blood_pressure = st.text_input("Enter your blood pressure (e.g., 120/80):")
            sleep = st.number_input("Enter hours of sleep:", min_value=0, max_value=24)
            exercise = st.number_input("Enter hours of exercise:", min_value=0, max_value=24)

            submitted = st.form_submit_button("Submit")

            if submitted:
                new_record = pd.DataFrame({
                'Date': [date],
                'Symptoms': [symptoms],
                'Weight': [weight],
                'Blood Pressure': [blood_pressure],
                'Sleep': [sleep],
                'Exercise': [exercise]
                })
                health_data = pd.concat([health_data, new_record], ignore_index=True)
                health_data.to_csv(DATA_FILE, index=False)
                st.success("Data saved!")

# Compare current week with previous week
        if len(health_data) >= 2:
            current_week_data = health_data.iloc[-1]
            previous_week_data = health_data.iloc[-2]

            st.subheader("Comparison with Previous Week")
            st.write(f"*Symptoms:* {current_week_data['Symptoms']} vs {previous_week_data['Symptoms']}")
            st.write(f"*Weight:* {current_week_data['Weight']} kg vs {previous_week_data['Weight']} kg")
            st.write(f"*Blood Pressure:* {current_week_data['Blood Pressure']} vs {previous_week_data['Blood Pressure']}")
            st.write(f"*Sleep:* {current_week_data['Sleep']} hours vs {previous_week_data['Sleep']} hours")
            st.write(f"*Exercise:* {current_week_data['Exercise']} hours vs {previous_week_data['Exercise']} hours")

    # Generate health precautions
        precautions = []
        if current_week_data['Weight'] > previous_week_data['Weight']:
            precautions.append("Consider monitoring your diet as weight has increased.")
        if current_week_data['Sleep'] < previous_week_data['Sleep']:
            precautions.append("Aim to improve your sleep duration for better health.")
        if "fever" in current_week_data['Symptoms'].lower():
            precautions.append("If you have a fever, consider seeing a doctor.")

        if precautions:
            st.subheader("Health Precautions:")
            for precaution in precautions:
                st.write(f"- {precaution}")

        # Email notification section
            email = st.text_input("Enter your email for health updates:")
            if st.button("Send Email"):
                send_email("Weekly Health Update", "\n".join(precautions), email)

        else:
            st.warning("Please enter at least two weeks of data to compare.")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
if __name__ == "__main__":
    main()

st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.markdown("<p style = 'color:grey;'>This is a prediction web app for educational purposes only.\n It is not a substitute for professional medical advice./nPlease consult a doctor or visit a hospital for proper diagnosis and treatment.</p>",unsafe_allow_html=True)
st.write("\n")
st.write("\n")
st.markdown('<p style="font-size:12px; color:#808080;">©2023 Internship Project </p>', unsafe_allow_html=True)




