import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle


def XGB_Model(genre,age,smoke,peer,chronic,alchool):
    xgb = pickle.load(open('model.pkl', 'rb'))
    # print(xgb.predict(X_test))
    DF = pd.DataFrame(columns=['GENDER',
                               'AGE',
                               'SMOKING',
                               'PEER_PRESSURE',
                               'CHRONIC_DISEASE'
        , 'ALCOHOL_CONSUMING'
                               ])
    print(type(DF.GENDER))
    DF.GENDER=DF.AGE=DF.SMOKING=DF.PEER_PRESSURE=DF.CHRONIC_DISEASE=DF.ALCOHOL_CONSUMING=0
    DF.loc[0,'GENDER']=genre
    DF.loc[0,'AGE']=age
    DF.loc[0,'SMOKING']=smoke
    DF.loc[0,'PEER_PRESSURE']=peer
    DF.loc[0,'CHRONIC_DISEASE']=chronic
    DF.loc[0,'ALCOHOL_CONSUMING']=alchool
    output=xgb.predict(DF)
    print(type(DF.GENDER), DF.GENDER)
    print(DF)
    print(xgb.score(DF, output))
    return output



print("predect; ",XGB_Model(1,66,2,2,1,2))