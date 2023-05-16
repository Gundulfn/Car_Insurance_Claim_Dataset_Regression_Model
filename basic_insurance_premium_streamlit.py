import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from category_encoders import TargetEncoder
import pickle

st.set_page_config(page_title="Base Insurance Premium Calculation Model")

st.write("## Car Insurance Claim Data - Regression Analysis")
df = pd.read_csv(r"autodata.csv")

for i in df.columns:
    if df[i].dtypes == "object":
        df[i] = df[i].str.replace('$' , '')
        df[i] = df[i].str.replace(',' , '')
        df[i] = df[i].str.replace("z_","")
        try:
            df[i] = df[i].astype(float)
        except:
            pass


with st.sidebar:
    add_radio = st.radio(
        "Please Choose A Process.",
        ("Data Preview", "Base Insurance Premium Calculation"))



if add_radio == "Data Preview":
    a = st.radio("##### Please Choose", ("Head", "Tail"))
    if a == "Head":
        st.table(df.head())
        
    if a == "Tail":
        st.table(df.tail())

    
    option = st.selectbox(
        '### Please Choose A Variable That You Want to Examine',
        df.columns.to_list())
    
    arr = df[option]
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)

    st.write("### The Columns That Won't Evaluate To Improve The Results:")

    st.write('drop some columns because of High Cardinality ("INCOME", "HOME_VAL") and Imbalance ("KIDSDRIV") and High Cardinality and Imbalance ("OLDCLAIM") and Unnecessary Column ("ID") and Overfitting ("CLM_AMT","CLAIM_FLAG") after severity and frequency calculation')
    
    st.write("#### Feature Importance Graphic for Frequency Model")
    image = Image.open(r"C:\Users\Asus\Desktop\kodluyoruz\hafta_4\feature_importance_frq.png")
    st.image(image ,width=800)
    st.write("#### Feature Importance Graphic for Severity Model")
    image = Image.open(r"C:\Users\Asus\Desktop\kodluyoruz\hafta_4\feature_importance_sev.png")
    st.image(image ,width=800)
    
if add_radio == "Base Insurance Premium Calculation":
    
    df = df[df.CLAIM_FLAG == 1]

    variables_list=["AGE","HOMEKIDS","YOJ","PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","TRAVTIME","CAR_USE","BLUEBOOK","TIF","CAR_TYPE",
                    "RED_CAR","CLM_FREQ","REVOKED","MVR_PTS","CAR_AGE"]

    box_desc_list = ['Parent is alive or not','Marital status', 'Male/Female', 'Degree holds by the customer', 'Job title',
                     'Purpose of the car (private/commercial)', 'Type of car (SUV/Pick up)', 'Colour of the car', 'Claim revoked']
                     
      
    slider_desc_list = ['Customers age', "Number of kids in the house", "Year of joining of the customer (employee/unemployee)",
                        'Travelling time', 'Legal citiation system in the United States', 'TIF',
                        'Number of times claims taken', 'Claim points', 'Age of the car']
                        

                        

    box_list = []
    slider_list = []

   
    for var in range(len(variables_list)):
        if df[variables_list[var]].dtype == "object":
            box_list.append(variables_list[var])
        elif df[variables_list[var]].dtype != "object":
            slider_list.append(variables_list[var])

    box_overall_dict = {}
    slider_overall_dict = {}

    # Creating dictionary for value names and their descriptions
    for var1, var2 in zip(box_list, box_desc_list):
        box_overall_dict.update({var1: var2})

    for var1, var2 in zip(slider_list, slider_desc_list):
        slider_overall_dict.update({var1: var2})

    # Displaying box and slider with functions
    def showing_box(var, desc):
            cycle_option = list(df[var].unique())#
            box = st.sidebar.selectbox(label= f"{desc}", options=cycle_option)
            return box

    def showing_slider(var, desc):
            slider = st.sidebar.slider(label= f"{desc}", min_value=round(df[var].min()), max_value=round(df[var].max()))
            return slider


    # Collecting user inputs in dictionaries
    box_dict = {}
    slider_dict = {}

    for key, value in box_overall_dict.items():
        box_dict.update({key: showing_box(key, value)})

    for key, value in slider_overall_dict.items():
        slider_dict.update({key: showing_slider(key, value)})


    # Keeping inputs in a dic
    input_dict = {**box_dict, **slider_dict}
    dictf = pd.DataFrame(input_dict, index=[0])
    #df = df.append(dictf, ignore_index= True) 
    df = pd.concat([df, dictf], ignore_index=True)

    
    df2 = df.copy()
        
    # High Cardinality
    df2.drop(labels=["INCOME", "HOME_VAL"],axis=1,inplace=True)

    # Imbalance
    df2.drop(labels=["KIDSDRIV"],axis=1,inplace=True)

    # High Cardinality and Imbalance
    df2.drop(labels=["OLDCLAIM"],axis=1,inplace=True)     

    df2.drop(["ID","CLM_AMT","CLAIM_FLAG"],axis=1,inplace=True)

    df_sev = df2.copy()
    df_frq = df2.copy()
    
    target_sev = open(r"Target_Encoder_sev.sav", 'rb')
    target_encoder_sev = pd.read_pickle(target_sev)
    #target_encoder = pickle.load(open(r"Target_Encoder.sav", 'rb'))
    
    df3_sev = pd.DataFrame(target_encoder_sev.transform(df_sev),index = df_sev.index,columns = df_sev.columns)

    # Selecting only last row. (User input data)
    newdata_sev =pd.DataFrame(df3_sev.iloc[[-1]])

    # Load already trained model (XGBoost)
    
    model_sev = open(r"SEV_Model.sav", 'rb')
    lr_sev = pd.read_pickle(model_sev)
    #lr = pickle.load(open(r"regression_model.sav", 'rb'))
    
    ypred_sev = lr_sev.predict(newdata_sev)
    
    # ----------------------------------------------------------------------------------------------------------
    
    target_frq = open(r"Target_Encoder_frq.sav", 'rb')
    target_encoder_frq = pd.read_pickle(target_frq)
    
    df3_frq = pd.DataFrame(target_encoder_frq.transform(df_frq),index = df_frq.index,columns = df_frq.columns)

    # Selecting only last row. (User input data)
    newdata_frq =pd.DataFrame(df3_frq.iloc[[-1]])

    # Load already trained model (XGBoost)
    
    model_frq = open(r"FRQ_Model.sav", 'rb')
    lr_frq = pd.read_pickle(model_frq)
    
    ypred_frq = lr_sev.predict(newdata_frq)
    
    ypred = ypred_sev * ypred_frq
    
    st.write("### Severity Result:")
    st.title(str(np.round(ypred_sev[0]/10,3))+" $")
    st.write("### Frequency Result:")
    st.title(str(np.round(ypred_frq[0]/1000,3)))
    
    st.write("### Base Insurance Premium Calculation Result:")
    st.title(str(np.round(ypred[0]/10000,3))+" $")

    image = Image.open(r"car_insurance_image.jpg")
    st.image(image ,width=800)
    
    st.write("### The Results of XGBRegressor Model")
    
    st.write('#### Frequency Model Result')
    image = Image.open(r"frq_model_result.png")
    st.image(image ,width=800)
    
    st.write('#### Severity Model Result')
    image = Image.open(r"sev_model_result.png")
    st.image(image ,width=800)
    
    st.write('#### Base Insurance Premium Calculation Results')
    image = Image.open(r"base_premium_model_result.png")
    st.image(image ,width=800)
    image = Image.open(r"base_premium_model_result2.png")
    st.image(image ,width=800)
