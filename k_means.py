import pandas as pd
from sklearn.cluster import KMeans

iris = pd.read_csv("./data/train.csv")
train = iris.iloc[:, [0, 1, 2, 3]].values

iris = pd.read_csv("./data/test.csv")
test = iris.iloc[:, [0, 1, 2, 3]].values

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans = kmeans.fit(train)
y_kmeans = kmeans.predict(test)
target_names = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
with open("./data/predict.txt","w+") as file:
    file.write(str([target_names[x] for x in y_kmeans]).replace(",",""))
