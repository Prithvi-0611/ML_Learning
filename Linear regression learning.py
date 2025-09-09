import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('advertising.csv')
print(df.head())

l1 = (df['TV'].values - df['TV'].mean()) / df['TV'].std()
l2 = (df['Radio'].values - df['Radio'].mean()) / df['Radio'].std()
l3 = (df['Newspaper'].values - df['Newspaper'].mean()) / df['Newspaper'].std()
y = (df['Sales'].values - df['Sales'].mean()) / df['Sales'].std()

w1=w2=w3=0.0
b=0
a=0.01

rmse_list=[]

for j in range(1000):
    j1=0
    j2=0
    j3=0
    j4=0
    for i in range(200):
        x=np.array([l1[i],l2[i],l3[i]])
        w = np.array([w1, w2, w3])
        j1+=((np.dot(x,w)+b-y[i])*l1[i])
    for i in range(200):
        x=np.array([l1[i],l2[i],l3[i]])
        w = np.array([w1, w2, w3])
        j2+=((np.dot(x,w)+b-y[i])*l2[i])
    for i in range(200):
        x=np.array([l1[i],l2[i],l3[i]])
        w = np.array([w1, w2, w3])
        j3+=((np.dot(x,w)+b-y[i])*l3[i])
    for i in range(200):
        x=np.array([l1[i],l2[i],l3[i]])
        w = np.array([w1, w2, w3])
        j4+=((np.dot(x,w)+b-y[i]))
    w1-=(a/200)*j1
    w2-=(a/200)*j2
    w3-=(a/200)*j3
    b-=(a/200)*j4

    preds = []
    for i in range(200):
        x = np.array([l1[i], l2[i], l3[i]])
        w = np.array([w1, w2, w3])
        preds.append(np.dot(x, w) + b)

    preds = np.array(preds)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    rmse_list.append(rmse)

print(w1,w2,w3,b,rmse_list[-1])

p=[]
for i in range(len(y)):
    x = np.array([l1[i], l2[i], l3[i]])
    w = np.array([w1, w2, w3])
    pred = np.dot(x, w) + b
    p.append(pred)

plt.scatter(y, preds, alpha=0.7)
plt.plot([-2, 2], [-2, 2], color="red")  # perfect prediction line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

import matplotlib.pyplot as plt
plt.plot(rmse_list)
plt.title("RMSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.show()














