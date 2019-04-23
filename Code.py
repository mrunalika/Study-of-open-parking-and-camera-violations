#Import Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Import other libraries for data pre-processing of data datatype 
from numba.tests.test_array_constants import dt
from skimage.util import dt
from pandas.types import dt
from nose.ext import dt
from pandas.lib import to_datetime
from time import time
from datetime import datetime

#Read in the file
os.chdir('C:\\mkadam\\ProjectCode')
violation_data=pd.read_table('Open_Parking_and_Camera_Violations.csv', delimiter='\t',sep='\t')
df.head()
df.dtypes

df['Violation_Time']=pd.to_datetime(df['Violation_Time'])
df['Violation_Time'] = pd.to_datetime(df['Violation_Time'], format='%H:%M').dt.time
df['Violation_Time'] = df['Violation_Time'].astype(str)

# Binning Violation_time column 
def f(row):
    if row['Violation_Time'] >= "00:00:00" and row['Violation_Time'] < "07:00:00":
        val = "midnight"
    elif row['Violation_Time'] >= "07:00:00" and row['Violation_Time'] < "12:00:00":
        val = "morning"
    elif row['Violation_Time'] >= "12:00:00" and row['Violation_Time'] < "16:00:00":
        val = "afternoon"
    else:
        val = "evening"
    return val

df['C'] = df.apply(f, axis=1)

#Checking data types for the violation_data
violation_data.dtypes

#Strip '$' from the amount columns
violation_data['Fine Amount'] = violation_data['Fine Amount'].str.lstrip('$')
violation_data['Penalty Amount']=violation_data['Penalty Amount'].str.lstrip('$')
violation_data['Interest Amount']=violation_data['Interest Amount'].str.lstrip('$')
violation_data['Reduction Amount']=violation_data['Reduction Amount'].str.lstrip('$')
violation_data['Payment Amount']=violation_data['Payment Amount'].str.lstrip('$')
violation_data['Amount Due']=violation_data['Amount Due'].str.lstrip('$')

violation_data.count()
violation_data.dtypes
violation_data.head()

#Handling Missing Values for continous variables
violation_data.iloc[363878,4]='7/20/2000'
violation_data.iloc[1008392,4]='6/02/2000'
violation_data.iloc[1008395,4]='6/02/2000'

#Making a copy of Violation data and doing Analysis on that
violation_data1= violation_data.copy()
violation_data1.drop(['Judgment Entry Date','Precinct','Violation Status'],axis=1,inplace=True)
violation_data1.dtypes

#Handling Missing values
violation_data1[['Penalty Amount','Interest Amount','Reduction Amount']] = violation_data1[['Penalty Amount','Interest Amount','Reduction Amount']].fillna(value=0)

#Count Na's
violation_data1.isnull().sum()
violation_data1 = violation_data1.dropna()

#Count Na's
violation_data1.isnull().sum()
violation_data1.count()

#Change Data Type for all Amount variables
violation_data1['Fine Amount']=violation_data1['Fine Amount'].astype(float)
violation_data1['Fine Amount']=violation_data1['Fine Amount'].astype(int)
violation_data1['Penalty Amount']=violation_data1['Penalty Amount'].astype(float)
violation_data1['Penalty Amount']=violation_data1['Penalty Amount'].astype(int)
violation_data1['Interest Amount']=violation_data1['Interest Amount'].astype(float)
violation_data1['Interest Amount']=violation_data1['Interest Amount'].astype(int)
violation_data1['Reduction Amount']=violation_data1['Reduction Amount'].astype(float)
violation_data1['Reduction Amount']=violation_data1['Reduction Amount'].astype(int)
violation_data1['Penalty Amount']=violation_data1['Penalty Amount'].astype(float)
violation_data1['Penalty Amount']=violation_data1['Penalty Amount'].astype(int)
violation_data1['Payment Amount']=violation_data1['Payment Amount'].astype(float)
violation_data1['Payment Amount']=violation_data1['Payment Amount'].astype(int)
violation_data1['Amount Due']=violation_data1['Amount Due'].astype(float)
violation_data1['Amount Due']=violation_data1['Amount Due'].astype(int)

#Renaming Columns for simplicity
violation_data1.rename(columns={'License Type':'License_Type'},inplace=True)
violation_data1.rename(columns={'Summons Number':'Summons_Number'},inplace=True)
violation_data1.rename(columns={'Issue Date':'Issue_Date'},inplace=True)
violation_data1.rename(columns={'Violation Time':'Violation_Time'},inplace=True)
violation_data1.rename(columns={'Judgement Entry Date':'Judgement_Entry_Date'},inplace=True)
violation_data1.rename(columns={'Fine Amount':'Fine_Amount'},inplace=True)
violation_data1.rename(columns={'Penalty Amount':'Penalty_Amount'},inplace=True)
violation_data1.rename(columns={'Interest Amount':'Interest_Amount'},inplace=True)
violation_data1.rename(columns={'Reduction Amount':'Reduction_Amount'},inplace=True)
violation_data1.rename(columns={'Payment Amount':'Payment_Amount'},inplace=True)
violation_data1.rename(columns={'Amount Due':'Amount_Due'},inplace=True)
violation_data1.rename(columns={'Issuing Agency':'Issuing_Agency'},inplace=True)
violation_data1.rename(columns={'Violation Status':'Violation_Status'},inplace=True)

#Erroneous data Handling to be done after removing all the missing values
violation_data1['State'].unique() #Has 1 erroneous data - 99
violation_data1['License_Type'].unique() #Has 1 erroneous data - 999
violation_data1['Issue_Date'] = violation_data['Issue_Date']=pd.to_datetime(violation_data['Issue Date'], format='%m/%d/%Y') # Handled Erroneous Data by converting it into DateTime format and removing other invalid years like 2099,2098 etc.

#Before Removal of erroneous data count
violation_data1.count()

#Removing erroneous data for both State and License Type
violation_data1 = violation_data1[violation_data1.State != '99']
violation_data1 = violation_data1[violation_data1.License_Type != '999']

#Identify and remove erroneous data for Issue Date
ts=pd.to_datetime('12/31/2017')
violation_data1.loc[violation_data1.Issue_Date>ts,:].count()
violation_data1 = violation_data1[violation_data1.Issue_Date <= ts]

#Appending 'M' to the Violation Time
violation_data1['Violation_Time']= violation_data1['Violation_Time'].astype(str) + 'M'

#Eliminating Erroneous data for Violation time
violation_data1 = violation_data1[violation_data1.Violation_Time != '2:30M']
violation_data1 = violation_data1[violation_data1.Violation_Time != '4:37M']
violation_data1 = violation_data1[violation_data1.Violation_Time != '6:00M']
violation_data1 = violation_data1[violation_data1.Violation_Time != '37:30PM']
violation_data1 = violation_data1[violation_data1.Violation_Time != '14:36PM']
violation_data1 = violation_data1[violation_data1.Violation_Time != '18:59PM']
violation_data1 = violation_data1[violation_data1.Violation_Time != '14:41PM']
violation_data1 = violation_data1[violation_data1.Violation_Time != '08:+0AM']

#After Removal of erroneous data count - 993976
violation_data1.count()
violation_data2 = violation_data1.copy()
violation_data2.Violation.value_counts()
violation_data2.Violation = violation_data1.Violation.replace(dict1)

dict1 = {"ALTERING INTERCITY BUS PERMIT":"other",
"ANGLE PARKING":"parking",
"ANGLE PARKING-COMM VEHICLE":"parking",
"BEYOND MARKED SPACE":"parking",
"BIKE LANE":"traffic",
"BUS LANE VIOLATION":"parking",
"BUS PARKING IN LOWER MANHATTAN":"parking",
"COMML PLATES-UNALTERED VEHICLE":"plate",
"CROSSWALK":"other",
"DETACHED TRAILER":"traffic",
"DIVIDED HIGHWAY":"traffic",
"DOUBLE PARKING":"parking",
"DOUBLE PARKING-MIDTOWN COMML":"parking",
"ELEVATED/DIVIDED HIGHWAY/TUNNL":"traffic",
"EXCAVATION-VEHICLE OBSTR TRAFF":"traffic",
"EXPIRED METER":"meter",
"EXPIRED METER-COMM METER ZONE":"meter",
"EXPIRED MUNI METER":"meter",
"EXPIRED MUNI MTR-COMM MTR ZN":"meter",
"FAIL TO DISP. MUNI METER RECPT":"meter",
"FAIL TO DSPLY MUNI METER RECPT":"meter",
"FAILURE TO DISPLAY BUS PERMIT":"other",
"FAILURE TO STOP AT RED LIGHT":"traffic",
"FEEDING METER":"meter",
"FIRE HYDRANT":"parking",
"FRONT OR BACK PLATE MISSING":"plate",
"IDLING":"no standing",
"IMPROPER REGISTRATION":"plate",
"INSP STICKER-MUTILATED/C'FEIT":"plate",
"INSP. STICKER-EXPIRED/MISSING":"plate",
"INTERSECTION":"traffic",
"MARGINAL STREET/WATER FRONT":"other",
"MISSING EQUIPMENT":"plate",
"NGHT PKG ON RESID STR-COMM VEH":"no standing",
"NIGHTTIME STD/ PKG IN A PARK":"no standing",
"NO MATCH-PLATE/STICKER":"plate",
"NO OPERATOR NAM/ADD/PH DISPLAY":"other",
"NO PARKING-DAY/TIME LIMITS":"parking",
"NO PARKING-EXC. AUTH. VEHICLE":"parking",
"NO PARKING-EXC. HNDICAP PERMIT":"parking",
"NO PARKING-EXC. HOTEL LOADING":"parking",
"NO PARKING-STREET CLEANING":"parking",
"NO PARKING-TAXI STAND":"parking",
"NO STANDING EXCP D/S":"no standing",
"NO STANDING EXCP DP":"no standing",
"NO STANDING-BUS LANE":"no standing",
"NO STANDING-BUS STOP":"no standing",
"NO STANDING-COMM METER ZONE":"no standing",
"NO STANDING-COMMUTER VAN STOP":"no standing",
"NO STANDING-DAY/TIME LIMITS":"no standing",
"NO STANDING-EXC. AUTH. VEHICLE":"no standing",
"NO STANDING-EXC. TRUCK LOADING":"no standing",
"NO STANDING-FOR HIRE VEH STOP":"no standing",
"NO STANDING-HOTEL LOADING":"no standing",
"NO STANDING-TAXI STAND":"no standing",
"NO STD(EXC TRKS/GMTDST NO-TRK)":"no standing",
"NO STOP/STANDNG EXCEPT PAS P/U":"no standing",
"NO STOPPING-DAY/TIME LIMITS":"no standing",
"NON-COMPLIANCE W/ POSTED SIGN":"other",
"OBSTRUCTING DRIVEWAY":"traffic",
"OBSTRUCTING TRAFFIC/INTERSECT":"traffic",
"OT PARKING-MISSING/BROKEN METR":"parking",
"OTHER":"other",
"OVERNIGHT TRACTOR TRAILER PKG":"other",
"OVERTIME PKG-TIME LIMIT POSTED":"meter",
"OVERTIME STANDING DP":"meter",
"PARKED BUS-EXC. DESIG. AREA":"parking",
"PEDESTRIAN RAMP":"parking",
"PHTO SCHOOL ZN SPEED VIOLATION":"traffic",
"PLTFRM LFTS LWRD POS COMM VEH":"other",
"RAILROAD CROSSING" :"traffic",
"REG STICKER-MUTILATED/C'FEIT":"plate",
"REG. STICKER-EXPIRED/MISSING":"plate",
"REMOVE/REPLACE FLAT TIRE":"no standing",
"SAFETY ZONE":"parking",
"SELLING/OFFERING MCHNDSE-METER":"meter",
"SIDEWALK":"other",
"STORAGE-3HR COMMERCIAL":"other",
"TRAFFIC LANE":"traffic",
"TUNNEL/ELEVATED/ROADWAY":"other",
"UNAUTHORIZED BUS LAYOVER":"no standing",
"UNAUTHORIZED PASSENGER PICK-UP":"no standing",
"VACANT LOT":"other",
"VEHICLE FOR SALE(DEALERS ONLY)":"other",
"VEH-SALE/WSHNG/RPRNG/DRIVEWAY":"other",
"VIN OBSCURED":"plate",
"WRONG WAY":"other"}

violation_data=violation_data1.copy()

violation_data.dtypes
violation_data.head()
X = violation_data.iloc[:, [7,11]].values
y = violation_data.iloc[:,6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train_summary = pd.DataFrame(X_train)
X_test_summary = pd.DataFrame(X_test)
y_train_summary = pd.DataFrame(y_train)
y_test_summary = pd.DataFrame(y_test)
X_train_summary.describe()
X_test_summary.describe()
y_train_summary.describe()
y_test_summary.describe()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-------------------------Decision Tree Model---------------------------------------
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(classifier, X, y, cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)

col_names = list(violation_data.ix[:,(7,11)].columns.values)
classnames = list(violation_data.Violation.unique())

tre2 = tree.DecisionTreeClassifier().fit(violation_data.ix[:,(7,11)],violation_data.Violation)

#Exporting decision tree into pdf
violation_data = StringIO()
tree.export_graphviz(tre2, out_file=violation_data,
                     feature_names=col_names,
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(violation_data.getvalue())
graph.write_pdf('tree.pdf')

#Tree pruning 
tre4 = tree.DecisionTreeClassifier(min_samples_split=210,min_samples_leaf=210)
tre4.fit(violation_data.ix[:,(7,11)],violation_data.Violation)
violation_data = StringIO()
tree.export_graphviz(tre4, out_file=violation_data,
                     feature_names=col_names,
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(violation_data.getvalue())
graph.write_pdf('tree1.pdf')

#-------------------------Random Forest Model---------------------------------------
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(classifier, X, y, cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)

#---------------------------Neural Network Model-----------------------
#Fitting Neural Network to the Training set
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

#relu activation function
nnclass2 = MLPClassifier(activation='relu', solver='sgd',
                         hidden_layer_sizes=(100,100))
nnclass2.fit(X_train, y_train)
nnclass2_pred = nnclass2.predict(X_test)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(nnclass2, X, y, cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

cm = metrics.confusion_matrix(y_test, nnclass2_pred)
print(cm)
accuracy_score(y_test,nnclass2_pred)
print(classification_report(y_test,nnclass2_pred))


#logistic activation function
nnclass3 = MLPClassifier(activation='logistic', solver='sgd',
                         hidden_layer_sizes=(100,100))
nnclass3.fit(X_train, y_train)

nnclass3_pred = nnclass3.predict(X_test)

cm = metrics.confusion_matrix(y_test, nnclass3_pred)
print(cm)
accuracy_score(y_test,nnclass3_pred)
print(classification_report(y_test,nnclass3_pred))
