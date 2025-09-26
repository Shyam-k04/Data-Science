import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load(r"C:\Users\Runku\model_Deployment_claiments\logistic_regression_model.pkl")

st.title('🚢⚓Titanic Survival Prediction (Logistic Regression)')


st.sidebar.header('Passenger Features')

def user_input_features():
    Pclass = st.sidebar.slider('Pclass', 1, 3, 3)
    Name = st.sidebar.number_input('Name (Encoded)', value=0) # Assuming encoded name input
    Sex = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: 'female' if x == 0 else 'male') # 0 for female, 1 for male
    Age = st.sidebar.slider('Age', 0.42, 80.0, 30.0)
    SibSp = st.sidebar.slider('SibSp', 0.0, 8.0, 0.0)
    Parch = st.sidebar.slider('Parch', 0.0, 6.0, 0.0)
    Ticket = st.sidebar.number_input('Ticket (Encoded)', value=0) # Assuming encoded ticket input
    Fare = st.sidebar.slider('Fare', 0.0, 512.3292, 32.0)
    Embarked = st.sidebar.selectbox('Embarked', [0, 1, 2], format_func=lambda x: ['C', 'Q', 'S'][x]) # Assuming encoded Embarked

    data = {'Pclass': Pclass,
            'Name': Name,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Ticket': Ticket,
            'Fare': Fare,
            'Embarked': Embarked}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Ensure the order of columns matches the training data
# Replace with the actual columns used in your training data
# You can get the column order from X.columns after training
expected_columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
input_df = input_df[expected_columns]


prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
survival_status = 'Survived' if prediction[0] == 1 else 'Not Survived'
st.write(f'The passenger is predicted to be: **{survival_status}**')

st.subheader('Prediction Probability')
st.write(f'Probability of Not Surviving: {prediction_proba[0][0]:.4f}')
st.write(f'Probability of Surviving: {prediction_proba[0][1]:.4f}')