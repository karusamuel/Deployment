import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("./model/model.joblib")

st.title("MPG predictor")

st.write("This is a simple app to predeict MPG")


displacement = st.number_input("Displacement",min_value=0.0)
weight = st.number_input("Weight",min_value=0.0)
horsepower = st.number_input("Horsepower",min_value=0.0)
acceleration = st.number_input("Acc",min_value=0.0)


if st.button("Predict") :
    
    data = {
        "displacement":displacement,
        "weight":weight,
        "horsepower":horsepower,
        "acceleration":acceleration
    }
    
    
    df = pd.DataFrame([data])
    
    pred = model.predict(df)
    
    st.write(f"Predicted MPG is {pred.tolist()[0]}")