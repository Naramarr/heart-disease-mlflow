from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("heart.csv")

df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

X = df.drop("target", axis =1)
y = df["target"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,random_state=43, test_size=0.2)

log = LogisticRegression()
log.fit(X_train, y_train)

y_pred = log.predict(X_test)

acc = accuracy_score(y_test,y_pred)
roc = roc_auc_score(y_test,y_pred)

print("report of both acc and roc are ",acc, roc)

joblib.dump(log,"model.pkl")
