# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start by importing the required libraries (pandas, matplotlib.pyplot, KMeans from sklearn.cluster).

2.Load the Mall_Customers.csv dataset into a DataFrame.

3.Check for missing values in the dataset to ensure data quality.

4.Select the features Annual Income (k$) and Spending Score (1-100) for clustering.

5.Use the Elbow Method by running KMeans for cluster counts from 1 to 10 and record the Within-Cluster Sum of Squares (WCSS).

6.Plot the WCSS values against the number of clusters to determine the optimal number of clusters (elbow point).

7.Fit the KMeans model to the selected features using the chosen number of clusters (e.g., 5).

8.Predict the cluster label for each data point and assign it to a new column called cluster.

9.Split the dataset into separate clusters based on the predicted labels.

10.Visualize the clusters using a scatter plot, and optionally mark the cluster centroids. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sanjay Sivaramakrishnan M
RegisterNumber:  212223240151

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\Mall_Customers.csv')
df.head()
df.info()
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('No. of  clusters')
plt.ylabel('wcss')
plt.title('Elbow method')
plt.show()
km = KMeans(n_clusters=5)
km.fit(df.iloc[:,3:])
y_pred = km.predict(df.iloc[:,3:])
y_pred
df['clusters'] = y_pred
df0 = df[df['clusters'] == 0]
df1 = df[df['clusters'] == 1]
df2 = df[df['clusters'] == 2]
df3 = df[df['clusters'] == 3]
df4 = df[df['clusters'] == 4]
df0
plt.scatter(df0['Annual Income (k$)'],df0['Spending Score (1-100)'])
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'])
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'])
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'])
plt.scatter(df4['Annual Income (k$)'],df4['Spending Score (1-100)'])


*/
```

## Output:
![image](https://github.com/user-attachments/assets/838d36ca-dffc-444c-9d5a-fd63d8cf1826)
![image](https://github.com/user-attachments/assets/98b15764-b8eb-4318-82d6-3fda16aec135)
![image](https://github.com/user-attachments/assets/dc046ff8-d387-4994-9fd3-e5bb8f56dd03)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
