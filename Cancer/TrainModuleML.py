import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_raw = pd.read_csv('survey lung cancer.csv')
df = df_raw.copy()
#df.info()
#elimine les ligne deplicate
df = df[~df.duplicated()]
#print(df)
#Changing Value for Gender Column Male : 1, Female : 0

df['GENDER'] = df['GENDER'].replace({'M' : 1, 'F' : 0})
df['GENDER'].value_counts()

key_rev = {'YES' : 1, 'NO' : 0}
df = df.replace(key_rev)
data= df
data= data.drop('YELLOW_FINGERS', axis = 1)
data= data.drop('ANXIETY', axis = 1)

data= data.drop('ALLERGY ', axis = 1)
data= data.drop('FATIGUE ', axis = 1)
data= data.drop('WHEEZING', axis = 1)
data= data.drop('COUGHING', axis = 1)
data= data.drop('SHORTNESS OF BREATH', axis = 1)
data= data.drop('SWALLOWING DIFFICULTY', axis = 1)
data= data.drop('CHEST PAIN', axis = 1)

X = data.drop('LUNG_CANCER', axis = 1)
y = data['LUNG_CANCER']

#print(df)
over_samp =  RandomOverSampler(random_state=0)
X_train_res, y_train_res = over_samp.fit_resample(X, y)
#X_train_res.shape, y_train_res.shape
X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size = 0.25, random_state = 42)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
XGBClassifierScore = xgb.score(X_test,y_test)
print("Précision obtenue par XGB Classifier model:",XGBClassifierScore*100)
# X_test.to_pickle("file_name")
# xgb = pd.read_pickle("file_name")

pickle.dump(xgb, open('model.pkl', 'wb'))
# xgb = pickle.load(open('model.pkl', 'rb'))
print(xgb.predict(X_test))
# DF = pd.DataFrame(columns=['GENDER',
#                            'AGE',
#                            'SMOKING',
#                            'PEER_PRESSURE',
#                            'CHRONIC_DISEASE'
#     , 'ALCOHOL_CONSUMING'
#                            ])
# print(type(DF.GENDER))
# DF.GENDER=DF.AGE=DF.SMOKING=DF.PEER_PRESSURE=DF.CHRONIC_DISEASE=DF.ALCOHOL_CONSUMING=0
# DF.loc[0,'GENDER']=int(1)
# DF.loc[0,'AGE']=int(55)
# DF.loc[0,'SMOKING']=int(2)
# DF.loc[0,'PEER_PRESSURE']=int(2)
# DF.loc[0,'CHRONIC_DISEASE']=int(2)
# DF.loc[0,'ALCOHOL_CONSUMING']=int(2)
# output=xgb.predict(DF)
#
# print(type(DF.GENDER),DF.GENDER)
# print(DF)
# print(xgb.score(DF,output))
# print("predect; ",output)
print(X_test)
print(X)





