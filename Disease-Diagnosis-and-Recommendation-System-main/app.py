import streamlit as st
import json
import hashlib
import os

# Define the path for the JSON file to store user data
USER_DATA_FILE = 'users.json'

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

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Streamlit UI
st.title("Login and Registration System")

# Only show login and registration options if not logged in
if not st.session_state['logged_in']:
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
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Incorrect username or password.")
else:
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    st.sidebar.write("You now have access to the sidebar functions.")
    
    # Display main application content here
    st.write("This is your main app content after login.")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.sidebar.success("Logged out successfully.")
