import json
import os
import pandas as pd

df = pd.read_csv('./data/test.csv')
f = open("./data/predict.txt", "r")
data = f.read().split()
f.close()
target = list(df['species'])

for i in range(len(data)):
	data[i] = data[i].replace("[", "").replace("]", "").replace("'", "")

#print(data)
#print(data)
#print(target)

acc = 0

for i in range(len(data)):
	if target[i] == data[i]:
		acc += 1

acc /= len(data)
acc *= 100
print(acc)
f = open('./data/eval.json', 'w+')
json.dump({'accuracy': acc}, f)
f.close()
