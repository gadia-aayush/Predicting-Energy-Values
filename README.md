## README- Predicting-Energy-Values

### **BRIEF DESCRIPTION:**

  - Here we are basically doing Time Series Forecasting of May month by using ARIMA Model.
  - Here I used a modified version of the Dataset called 'final.csv'

-------------------------------------------------------------------------------------------------------------------


### **PREREQUISITES:**

  - written in  Python 3.6 .

-------------------------------------------------------------------------------------------------------------------


### **CLIENT-END FULFILMENTS:**

**Steps to predict:**

  1. ***Run- '1_final_data-loading.py' file.***   
    - By running it actually we will consider April & May values for modelling our problem & May values for validating our prediction.  
    - 2 files will be created final_dataset.csv & final_validation.csv  

  2. ***Run- '2_final_problem-evaluate.py'***  
    - We will evaluate the problem by Baseline prediction & by Drawing various Plots.  
    - Output given in jpeg.  

  3. ***Run- '3_final_arima-modelling.py'***  
    - Now we will implement ARIMA Model on the Dataset.  
    - Here the program will automatically find the best (p,d,q) values for the Problem.  
    - Output given in jpeg.  

  4. ***Run- '4_final_arima-residualerrors.py'***  
    - Here checking for Residual Errors to see whether Stationarity in Dataset is acheived or not.  
    - Output given in jpeg.  

  5. ***Run- '5_final_model-save.py'***  
    - Now we will implement the model on entire dataset & save it as 'final_model.pkl' & to use it for future 
      predictions.  
    - Output given in jpeg.


  6. ***Run- '6_final_predict.py'***  
    - Now this will finally predict the values of the May month along with what was Expected & \
      what it predicted.  
    - Output given in jpeg.  


-------------------------------------------------------------------------------------------------------------------	

### **OUTPUT SAMPLE:**  
  -	Please refer the Output Screenshots Folder.
  

-------------------------------------------------------------------------------------------------------------------	

### **AUTHORS:**  

  -	coded by AAYUSH GADIA.

   
					  

