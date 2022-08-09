import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier_dt = pickle.load(open('churndt.pkl','rb'))
classifier_knn = pickle.load(open('churnknn.pkl','rb'))
classifier_svm = pickle.load(open('churnsvm.pkl','rb'))
classifier_rf = pickle.load(open('churnrf.pkl','rb'))
classifier_NB = pickle.load(open('churnnb.pkl','rb'))

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    credit = int(request.args.get('credit'))
    contry = (request.args.get('contry'))

    if contry=="France":
      contry = 0
    elif contry=="Spain":
      contry = 1
    else:
      contry = 2


    gender = (request.args.get('gender'))
    
    if gender=="Male":
      gender = 1
    else:
      gender = 0

    age = int(request.args.get('age'))
    ten = int(request.args.get('ten'))
    balance = float(request.args.get('balance'))
    prodno = int(request.args.get('prodno'))

    creditsc = (request.args.get('creditsc'))

    if creditsc=="Yes":
      creditsc = 1
    else:
      creditsc = 0

    active = (request.args.get('active'))

    if active=="Yes":
      active = 1
    else:
      active = 0

    sal = float(request.args.get('sal'))
    

# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	
    Model = (request.args.get('Model'))

    if Model=="Random Forest":
      prediction = classifier_dt.predict([[credit, contry, gender, age, ten, balance, prodno, creditsc, active, sal]])

    elif Model=="Decision Tree":
      prediction = classifier_knn.predict([[credit, contry, gender, age, ten, balance, prodno, creditsc, active, sal]])

    elif Model=="KNN":
      prediction = classifier_svm.predict([[credit, contry, gender, age, ten, balance, prodno, creditsc, active, sal]])

    elif Model=="SVM":
      prediction = classifier_rf.predict([[credit, contry, gender, age, ten, balance, prodno, creditsc, active, sal]])

    else:
      prediction = classifier_NB.predict([[credit, contry, gender, age, ten, balance, prodno, creditsc, active, sal]])

    
    if prediction == [0]:
      return render_template('index.html', prediction_text='Person is Excited', extra_text =" as per Prediction by " + Model)
    
    else:
      return render_template('index.html', prediction_text='The Person is not excited', extra_text ="as per Prediction by " + Model)

if __name__=="__main__"
app.run(debug=True)

