import pandas as pd
train = pd.read_csv('dataset.csv')
for i in range(len(train['Type'])):
	if(train['Type'][i]=='CAR'):
		train['continuousPixelRowWise'].at[i]=train['continuousPixelRowWise'].at[i]+10

train.to_csv("dataset.csv", sep=',', encoding='utf-8')