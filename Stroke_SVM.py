#  import libraries
import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#--------------#
#DATA PROCESSING
#--------------#

#  import data, check sets
data = pd.read_csv(r"C:\Users\Byon8\PycharmProjects\Stroke\stroke_train.csv")

values = {'smoking_status': "Unknown"}
data = data.fillna(value=values)

data = data.dropna()

x_pp = data.drop(['id', 'stroke'], axis = 1)
y = data['stroke']

#  label encode binary columns Residence_type and ever_married 
x_pp = x_pp.replace(
	["No", "Yes",
	"Rural", "Urban"], 
	[0, 1,
	0, 1]
)

#  OHE for gender, work_type, smoking_status
x_pp = pd.get_dummies(x_pp, columns = ['gender', 'work_type', 'smoking_status'])

#  rename columns as appropriate
x_pp = x_pp.rename(index=str, columns={"Residence_type": "urban_residence", 
	"work_type_Self-employed": "work_type_Self_employed",
	"smoking_status_formerly smoked": "smoking_status_formerly_smoked", 
	"smoking_status_never smoked": "smoking_status_never_smoked"})

#  scale numerical data
scaler = preprocessing.MinMaxScaler()
x = pd.DataFrame(scaler.fit_transform(x_pp), columns=x_pp.keys())

#  previous step converts all datatypes to floats, so convert the non-floats back to ints
integer_columns = ['hypertension', 'heart_disease', 'ever_married', 'urban_residence', 
	'gender_Female', 'gender_Male', 'work_type_Govt_job', 
	'work_type_Never_worked', 'work_type_Private', 'work_type_Self_employed', 
	'work_type_children', 'smoking_status_Unknown', 'smoking_status_formerly_smoked', 
	'smoking_status_never_smoked', 'smoking_status_smokes']

for col in integer_columns:
	x[col] = x[col].astype(int)

#  split into stratified 80% test, 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.20)  

#-------------------------#
#CREATE AND EVALUATE MODEL
#-------------------------#

#  create SVM model
svclassifier = SVC(kernel='linear') #  rbf, poly, sigmoid
svclassifier.fit(x_train, y_train)

prediction = svclassifier.predict(x_test)

#  precision, recall, f1, support
print(confusion_matrix(y_test,prediction))  
print(classification_report(y_test,prediction))

#  accuracy
print("Accuracy: {}%".format(svclassifier.score(x_test, y_test) * 100 ))
