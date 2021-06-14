from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn import svm
import pickle

df = pd.read_csv('dataset2.csv')
print(df)
label_encoder = 0
encoded_map = {}

for i,d in enumerate(df["LOCATION"]):
    if not str(d) in encoded_map:
        print(d)
        df["LOCATION"][i] = label_encoder
        encoded_map[str(d)] = label_encoder
        label_encoder += 1
    else:
        df["LOCATION"][i] = encoded_map[str(d)]

print(df)


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X= df.iloc[ 1: , :-1].values
print(X)
y= df.iloc[1:, -1:].values
print(y)
regr = svm.SVR()
r =regr.fit(X, y)
filename = "svm.sav"
pickle.dump(r, open(filename, 'wb'))
print(encoded_map)
