import streamlit as st
from helper import main1
from LLM_Integration import user_input
import pandas as pd

# Title of the app with icon
st.title("Mental Health Prediction App")
st.markdown("""
    This app predicts mental health conditions based on user input data. 
    Please fill in the details below and select the model(s) for prediction.
""")

# Organize input fields into sections with headers
st.header("Personal Details")

col1, col2 = st.columns(2)

with col1:
    phq_score = st.number_input("PHQ Score", min_value=0, max_value=27, value=12, help="The PHQ score helps identify depression severity.")
    depressiveness = st.selectbox("Depressiveness", ["True", "False"], help="Whether the user exhibits depression-like symptoms.")
    bmi = st.number_input("BMI", min_value=0.0, value=24.5, help="Body Mass Index (BMI) helps assess health risk.")
    epworth_score = st.number_input("Epworth Score", min_value=0, max_value=24, value=10, help="Epworth score to assess daytime sleepiness.")
    gad_score = st.number_input("GAD Score", min_value=0, max_value=21, value=15, help="Generalized Anxiety Disorder (GAD) score.")

with col2:
    depression_severity = st.selectbox(
        "Depression Severity", 
        ["None-minimal", "Mild", "Moderate", "Moderately severe", "Severe", "none"], 
        help="Severity level of depression."
    )
    age = st.number_input("Age", min_value=0, value=22, help="User's age.")
    school_year = st.number_input("School Year", min_value=1, value=4, help="User's current school year.")
    anxiety_severity = st.selectbox(
        "Anxiety Severity", 
        ["None-minimal", "Mild", "Moderate", "Severe", "0"], 
        help="Severity level of anxiety."
    )
    who_bmi = st.selectbox(
        "WHO BMI", 
        ["Normal", "Underweight", "Overweight", "Class I Obesity", "Class II Obesity", "Class III Obesity", "Not Available"], 
        help="WHO BMI category."
    )

# Model selection section with better styling
st.header("Select Models for Prediction")

model_options = st.multiselect(
    "Choose the model(s):",
    options=["Random Forest", "XGBoost"],
    default=["Random Forest", "XGBoost"],
    help="Select one or more models for prediction."
)

# Create a dictionary from the input data
input_data = {
    'phq_score': [phq_score],
    'depressiveness': [depressiveness],
    'bmi': [bmi],
    'epworth_score': [epworth_score],
    'gad_score': [gad_score],
    'depression_severity': [depression_severity],
    'age': [age],
    'school_year': [school_year],
    'anxiety_severity': [anxiety_severity],
    'who_bmi': [who_bmi]
}

# Convert the dictionary to a DataFrame
user_input_df = pd.DataFrame(input_data)
original_input_df = user_input_df.copy()

# Mapping categorical columns to their encoded values
depression_severity_mapping = {
    "None-minimal": 3,
    "Mild": 0,
    "Moderate": 1,
    "Moderately severe": 2,
    "Severe": 4,
    "none": 5
}
anxiety_severity_mapping = {
    "None-minimal": 3,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 4,
    "0": 0
}
who_bmi_mapping = {
    "Normal": 3,
    "Underweight": 6,
    "Overweight": 5,
    "Class I Obesity": 0,
    "Class II Obesity": 1,
    "Class III Obesity": 2,
    "Not Available": 4
}

# Apply mappings to categorical columns
user_input_df['depression_severity'] = user_input_df['depression_severity'].map(depression_severity_mapping)
user_input_df['anxiety_severity'] = user_input_df['anxiety_severity'].map(anxiety_severity_mapping)
user_input_df['who_bmi'] = user_input_df['who_bmi'].map(who_bmi_mapping)

# Map 'depressiveness' column to numeric values
user_input_df['depressiveness'] = user_input_df['depressiveness'].map({'True': 1, 'False': 0})

# Button to make prediction with style and message
if st.button("🔮 Predict"):
    if not user_input_df.empty and model_options:
        with st.spinner("Processing your prediction..."):
            # Call the main function from helper.py
            prediction = main("depression_anxiety_data.csv", user_input_df)

            # Filter predictions based on selected models
            filtered_prediction = {model: prediction[model] for model in model_options}

            # Display predictions with color-coded results
            st.subheader("📊 Prediction Results")
            for model, result in filtered_prediction.items():
                if result:
                    st.error(f"⚠️ {model}: Positive (Indicates a diagnosable condition)")
                else:
                    st.success(f"✅ {model}: Negative (No diagnosable condition detected)")

            # Call the user_input function from LLM_Integration.py
            response = user_input(original_input_df , prediction)

            # Display the response in a better format
            st.subheader("💬 Detailed Explanation and Recommendations")
            st.info(response)
    else:
        st.warning("⚠️ Please enter the data for prediction and select at least one model.")
