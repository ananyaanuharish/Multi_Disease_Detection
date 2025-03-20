import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models and scalers
diabetes_model = pickle.load(open('./voting_classifier_diabetes.sav', 'rb'))
heart_disease_model = pickle.load(open('./voting_classifier_heart.sav', 'rb'))
parkinsons_model = pickle.load(open('./voting_classifier_parkinsons.sav', 'rb'))

diabetes_scaler = pickle.load(open('./diabetes_scaler.sav', 'rb'))
heart_scaler = pickle.load(open('./heart_scaler.sav', 'rb'))
parkinsons_scaler = pickle.load(open('./parkinsons_scaler.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

st.title("Multiple Disease Prediction System")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.subheader('Diabetes Prediction')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            scaled_data = diabetes_scaler.transform([input_data])  # Scale the input data
            diab_prediction = diabetes_model.predict(scaled_data)
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = "Please enter valid numeric values."

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.subheader('Heart Disease Prediction')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
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
        thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    # Prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            scaled_data = heart_scaler.transform([input_data])  # Scale the input data
            heart_prediction = heart_disease_model.predict(scaled_data)
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        except ValueError:
            heart_diagnosis = "Please enter valid numeric values."

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.subheader("Parkinson's Disease Prediction")

    # Input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('Spread1')
    with col5:
        spread2 = st.text_input('Spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # Prediction
    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            input_data = [
                float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR),
                float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2),
                float(D2), float(PPE)
            ]
            scaled_data = parkinsons_scaler.transform([input_data])  # Scale the input data
            parkinsons_prediction = parkinsons_model.predict(scaled_data)
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "Please enter valid numeric values."

    st.success(parkinsons_diagnosis)
