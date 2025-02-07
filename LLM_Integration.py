import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from helper import main1
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd

# Configure Google API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def user_input(user_data , prediction):
    # Define the prompt template
    prompt_template = """
    Given the following user input data and predictions, provide natural language explanations for the predicted mental health conditions. Additionally, suggest coping mechanisms and potential next steps.
    Predictions: {Prediction}
    Data for prediction: {data}
    Explain in detail.
    Answer:
    """
    
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Create the prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["Prediction", "data"])
    
    # Create the LLMChain
    chain = LLMChain(llm=model, prompt=prompt)
    
    # Get predictions from the helper function
    # prediction = main("depression_anxiety_data.csv", user_data)
    
    # Run the chain with the input data
    response = chain.run(Prediction=prediction, data=user_data)
    
    return response

# # Example user input data
# user_input_data = {
#     'phq_score': [12],
#     'depressiveness': ['True'],
#     'bmi': [24.5],
#     'epworth_score': [10],
#     'gad_score': [15],
#     'depression_severity': [3],
#     'age': [22],
#     'school_year': [4],
#     'anxiety_severity': [2],
#     'who_bmi': [25]
# }

# # Convert user input data to DataFrame
# user_input_df = pd.DataFrame(user_input_data)
# response  = user_input(user_input_df)
# # Call the user_input function and print the response
# print(response)
