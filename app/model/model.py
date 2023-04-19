from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import pickle

import numpy as np # for data manipulation
import pandas as pd # for data manipulation
from sklearn.model_selection import train_test_split # will be used for data split
from sklearn.ensemble import RandomForestClassifier # for training the algorithm

#Leyendo archivo CSV mediante github raw
df = pd.read_csv('https://raw.githubusercontent.com/xChoki/FML-Datos-CSGO/main/datos.csv', sep=";")

# Transformando variable MatchWinner a valor numérico
df["MatchWinner"] = df["MatchWinner"].astype(int)

# Transformando variable Survived a valor numérico
df["Survived"] = df["Survived"].astype(int)

df = df.drop(['Unnamed: 0', 'RoundWinner', 'RoundKills','RoundAssists','RoundHeadshots','Map','Team','InternalTeamId','MatchId','RoundId', 'SteamId', 'AbnormalMatch', 'TimeAlive', 'ScaledTimeAlive', 'AvgCentroidDistance', 'TravelledDistance', 'AvgRoundVelocity', 'AvgKillDistance', 'AvgSiteDistance', 'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'FirstKillTime', 'FirstKillTime', 'RoundFlankKills', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue', 'MatchFlankKills', 'AvgMatchKillDist'], axis = 1)

y = df['MatchWinner']
X = df.drop('MatchWinner', axis = 1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, stratify=y)

#print(df)

#clf = RandomForestClassifier(n_estimators=1000)
clf = AdaBoostClassifier(n_estimators=5000)
print(clf.fit(Xtrain.values, Ytrain.values).score(Xtest.values, Ytest.values))

filename = 'checkpoints/model.pkl'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(Xtest.values, Ytest.values)
print(result)
print(loaded_model.predict([[2, 2, 2, 1]]))
#print(X)
#print(y)