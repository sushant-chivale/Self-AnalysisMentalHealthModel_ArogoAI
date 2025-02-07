# Self-Analysis Mental Health Model

## Overview
This project has been deployed on Streamlit and can be accessed at:
[Self-Analysis Mental Health Model](https://self-analysis-mental-health-model-arogoai.streamlit.app/)

This project develops a **Self-Analysis Mental Health Model** that predicts possible mental health conditions based on user-provided symptoms. The model utilizes machine learning algorithms (Random Forest & XGBoost) and integrates an **LLM-based explanation system** to provide insights and coping mechanisms. A **Streamlit-based UI** is implemented for user interaction.

## Dataset
The model is trained on the **Depression and Anxiety Data** dataset from Kaggle:
[Depression and Anxiety Data](https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data/data)

## Features
- **Machine Learning Models:** Random Forest & XGBoost trained to predict the `suicidal` column.
- **LLM Integration:** Provides explanations for predictions and suggests coping mechanisms.
- **Streamlit UI:** Allows users to input symptoms and receive predictions and explanations.
- **SHAP for Interpretability:** Feature importance and impact on predictions.

## File Structure
```
📂 Self-Analysis Mental Health Model
│── .streamlit/                   # Streamlit configuration files
│── .env                           # Environment variables
│── Finetuning_Model.py            # Model training and evaluation
│── LLM_ExperimentationReport.pdf  # Report on LLM integration
│── LLM_Integration.py             # LLM-based explanations
│── README.md                      # Project documentation
│── depression_anxiety_data.csv     # Original dataset
│── depression_anxiety_data_with_explanations.csv  # Dataset with explanations
│── helper.py                       # Helper functions for processing
│── mental_health_ui.py              # Streamlit UI implementation
│── predict_mental_health.ipynb      # Inference script for testing
│── requirements.txt                 # Dependencies
```

## Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ installed. 

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit UI
```bash
streamlit run mental_health_ui.py
```

### Run the Inference Script (CLI-Based)
```bash
python predict_mental_health.ipynb
```

## Model Training
To train and evaluate the model, run:
```bash
python Finetuning_Model.py
```

## LLM Integration
To generate explanations using LLM:
```bash
python LLM_Integration.py
```


## Video Demonstration
A walkthrough of the project, including:
- Code structure and workflow
- Model training and testing
- Sample predictions and explanations


## Contact
For any queries, feel free to reach out!
