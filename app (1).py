import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


model=pickle.load(open("model_svm.sav",'rb'))



st.title('Telecommunication Churn Prediction')

def churnprediction(intl_plan, voice_plan, day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge, night_calls, night_mins, night_charge):
    input=np.array([[intl_plan, voice_plan, day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge, night_calls, night_mins, night_charge]],dtype=object)
    prediction=model.predict(input)
    return prediction

def main():
    intl_plan=st.selectbox('International Plan', ('YES', 'NO'))
    voice_plan = st.selectbox('Voice Plan', ('YES', 'NO'))
    day_calls=st.number_input('No of Day Calls:',min_value=47,max_value=155)
    day_mins=st.number_input('Duration of Day Calls:',min_value=34.0,max_value=325.0)
    day_charge=st.number_input('Cost Charged for Day Calls:',min_value=5.5,max_value=52.5)
    eve_calls=st.number_input('No of Evening Calls:',min_value=46.5,max_value=155.0)
    eve_mins=st.number_input('Duration of Evening Calls:',min_value=65.0,max_value=335.0)
    eve_charge=st.number_input('Cost Charged for Evening Calls:',min_value=5.4,max_value=29.0)
    night_calls=st.number_input('No of Night Calls:',min_value=48,max_value=153)
    night_mins=st.number_input('Duration of Night Calls:',min_value=65.0,max_value=336.5)
    night_charge=st.number_input('Cost Charged for Night Calls:',min_value=2.8,max_value=15.5)
    
    le=LabelEncoder()
    intl_plan = le.fit_transform([intl_plan])[0]
    voice_plan = le.fit_transform([voice_plan])[0]
    
    if st.button("Predict"):
        output=churnprediction(intl_plan, voice_plan, day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge, night_calls, night_mins, night_charge)
        if output== 1:
            st.write('This customer is likely to churn')
        else:
            st.write('This customer will likely not churn')
if __name__=="__main__":
    main()