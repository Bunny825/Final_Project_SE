import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow import keras

heart_model=pickle.load(open("heart.sav","rb"))
diabetes_model=pickle.load(open("diabetes_model.sav","rb"))
parkinson_model=pickle.load(open("parkinson_model.sav","rb"))

cardio_model=tf.keras.models.load_model("cardio_model.h5")

charges_model=tf.keras.models.load_model("charges_model.h5")


charges_scaler=pickle.load(open("charges_scaler.sav","rb"))

parkinson_scaler=pickle.load(open("parkinson_scaler.sav","rb"))
diabetes_scaler=pickle.load(open("diabetes_scaler.sav","rb"))
cardio_scaler=pickle.load(open("cardio_scaler.sav","rb"))

with st.sidebar:
	selected=option_menu("ILAB",
											["Heart Disease Prediction","Diabetes Prediction","Parkinson Prediction","Cardio Diesase Prediction","Medical expenses"],
											icons=["heart","activity","person","balloon-heart-fill","cash"],
											default_index=0)
											
if selected=="Heart Disease Prediction":
	def pred_heart(inp):
		inp = np.asarray(inp)
		inp = inp.reshape(1, -1)
		ans = heart_model.predict(inp)
		if ans == 0:
		    return "No heart disease"
		else:
		    return "Clear signs of heart disease"
        
        
	st.title("Heart Disease Prediction")
    
	col1,col2,col3=st.columns(3)
    
    
    # Input fields
	with col1:
		age = st.text_input("Age")
	with col2:
		sex = st.text_input("Gender")
	with col3:
		cp = st.text_input("Chest Pain")    
	with col1:
		trestbps = st.text_input("Resting Blood Pressure")    
	with col2:
		chol = st.text_input("Serum Cholestrol")    
	with col3:
		fbs = st.text_input("Fasting Blood Sugar")
	with col1:
		restecg = st.text_input("Resting ECG Results")        
	with col2:
		thalach = st.text_input("Maximum Heart Rate Achieved")        
	with col3:
		exang = st.text_input("Exercise-induced Angina")        
	with col1:
		oldpeak = st.text_input("Oldpeak")        
	with col2:
		slope = st.text_input("Slope")
	with col3:
		ca = st.text_input("Number of Major Vessels (0-3) colored by fluoroscopy")
	with col1:
		thal = st.text_input("Thalassemia")
    
    # Prediction and output
	diagnosis = ""
	if st.button("Get Heart Disease Prediction"):
		input_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg),
                          int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]
		diagnosis = pred_heart(input_data)
    
	st.success(diagnosis)

if selected=="Diabetes Prediction":
	def pred_diabetes(inp):
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=diabetes_scaler.transform(inp)
		pred=diabetes_model.predict(inp)
		if pred[0]==0:
			return "The patient is non diabetic"
		else:
			return "The patient is diabetic"
    		
	st.title("Diabetes Prediction")
	
	col1,col2,col3=st.columns(3)
	with col1:
		Pregnancies=st.text_input("Pregnancies")	
	with col2:
		Glucose=st.text_input("Glucose")	
	with col3:
		BloodPressure=st.text_input("BP")	
	with col1:
		SkinThickness=st.text_input("SkinThickness")	
	with col2:
		Insulin=st.text_input("Insulin")	
	with col3:
		BMI=st.text_input("BMI")	
	with col1:
		DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function")	
	with col2:
		Age=st.text_input("Age")
	
	result=""
	if st.button("Diabetes Prediction"):
		inp=[	int(Pregnancies),	int(Glucose),int(	BloodPressure),  int(SkinThickness)	,int(Insulin),	float(BMI)	,float(DiabetesPedigreeFunction)	,int(Age)	]
		result=pred_diabetes(inp)
	st.success(result)
	
	
if selected=="Parkinson Prediction":
	st.title("Parkinsons Disease Prediction")
	
	def pred_parkinson(inp):
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=parkinson_scaler.transform(inp)
		ans=parkinson_model.predict(inp)

		if ans==0:
			return "The person is perfectly healthy"
		elif ans==1:
			return "The person is has Parkinsons Disease"
	
	
	col1,col2,col3=st.columns(3)
	with col1:
		MDVP_Fo_Hz = st.text_input("MDVP:Fo(Hz)")
	with col2:
		MDVP_Fhi_Hz =st.text_input("MDVP:Fhi(Hz)")
	with col3:
		MDVP_Flo_Hz =st.text_input("MDVP:Flo(Hz)")
	with col1:
		MDVP_Jitter_percent = st.text_input("MDVP:Jitter(%)")
	with col2:
		MDVP_Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
	with col3:
		MDVP_RAP = st.text_input("MDVP:RAP")
	with col1:
		MDVP_PPQ = st.text_input("MDVP:PPQ")
	with col2:
		Jitter_DDP = st.text_input("Jitter:DDP")
	with col3:
		MDVP_Shimmer = st.text_input("MDVP:Shimmer")
	with col1:
		MDVP_Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
	with col2:
		Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
	with col3:
		Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
	with col1:
		MDVP_APQ = st.text_input("MDVP:APQ")
	with col2:
		Shimmer_DDA = st.text_input("Shimmer:DDA")
	with col3:
		NHR = st.text_input("NHR")
	with col1:
		HNR = st.text_input("HNR")
	with col2:
		RPDE = st.text_input("RPDE")
	with col3:
		DFA = st.text_input("DFA")
	with col1:
		spread1 = st.text_input("spread1")
	with col2:
		spread2 = st.text_input("spread2")
	with col3:
		D2 = st.text_input("D2")
	with col1:
		PPE = st.text_input("PPE")


	answer=""
	if st.button("Parkinsons prediction"):
		inp=[float(MDVP_Fo_Hz),float(MDVP_Fhi_Hz), float(MDVP_Flo_Hz),float(MDVP_Jitter_percent),
		float(MDVP_Jitter_Abs),float(MDVP_RAP),
	  float(MDVP_PPQ),
	  float(Jitter_DDP),float(MDVP_Shimmer),float(MDVP_Shimmer_dB),
	  float(Shimmer_APQ3),float(Shimmer_APQ5),float(MDVP_APQ),
		float(Shimmer_DDA),float(NHR),float(HNR),float(RPDE),float(DFA),float(spread1),float(spread2),float(D2),float(PPE)]

		answer=pred_parkinson(inp)
		
	st.success(answer)
	
	
	
	
	
	
if selected=="Cardio Diesase Prediction":
	def pred_cardiovascular(inp):
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=cardio_scaler.transform(inp)
		pred=cardio_model.predict(inp)
		if pred[0]>0.5:
			return "The patient has cardiac disease"
		else:
			return "The patient has no cardiac disease"
    		
	st.title("Cardiac Disease prediction")
	
	col1,col2,col3=st.columns(3)
	with col1:
		Pregnancies=st.text_input("Age")	
	with col2:
		Glucose=st.text_input("Gender")	
	with col3:
		BloodPressure=st.text_input("Height")	
	with col1:
		SkinThickness=st.text_input("Weight")	
	with col2:
		Insulin=st.text_input("ap_hi")	
	with col3:
		BMI=st.text_input("ap_lo")	
	with col1:
		DiabetesPedigreeFunction=st.text_input("Cholesterol")	
	with col2:
		Age=st.text_input("Glucose")
	with col3:
		smoke=st.text_input("Smoke")	
	with col1:
		alcohol=st.text_input("Alcohol")	
	with col2:
		active=st.text_input("Active")
		
		
	result=""
	if st.button("cardiovasular prediction"):
		inp=[	int(Pregnancies),	int(Glucose),int(	BloodPressure),  int(SkinThickness)	,int(Insulin),	float(BMI)	,float(DiabetesPedigreeFunction)	,int(Age)	,int(smoke),int(alcohol),int(active)]
		result=pred_cardiovascular(inp)
	st.success(result)
	
	
	
	
if selected=="Medical expenses regression":
	def pred_medicalexpenses(inp):
		if inp[1]=="male":
			inp[1]=1
		elif inp[1]=="female":
			inp[1]=0
			
		if inp[4]=="yes":
			inp[4]=1
		elif inp[4]=="no":
			inp[4]=0
			
		if inp[5]=="southeast":
			inp[5]=1
		elif inp[5]=="southwest":
			inp[5]=2
		elif inp[5]=="northeast":
			inp[5]=4
		elif inp[5]=="northwest":
			inp[5]=3
			
		
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=charges_scaler.transform(inp)
		pred=charges_model.predict(inp)
		return pred
    		
	st.title("Medical Charges Prediction")
	
	col1,col2,col3=st.columns(3)
	with col1:
		age=st.text_input("Age")	
	with col2:
		sex=st.text_input("Sex")	
	with col3:
		bmi=st.text_input("BMI")	
	with col1:
		children=st.text_input("Children")	
	with col2:
		smoker=st.text_input("Smoker")	
	with col3:
		region=st.text_input("Region")	
	
		
		
	result=""
	if st.button("medical expenses prediction"):
		inp=[	int(age),sex,float(bmi),int(children)	,smoker,region]
		result=pred_medicalexpenses(inp)
	st.success(result)
