import os #to get working directory
import cv2
import numpy as np
import pandas as pd

def getRowAverage(image, threshold=250, averageThreshold=50):
	continuousPixel = []
	for i in range(image.shape[0]):
		continuousPixelCount=0
		for j in range(image.shape[1]):
			if image[i,j]>threshold:
				continuousPixelCount+=1
			else:
				continuousPixel.append(continuousPixelCount)
				continuousPixelCount=0
		continuousPixel.append(continuousPixelCount)

	sum=0
	count=0
	longest=-1
	for i in continuousPixel:
		if i > averageThreshold:
			sum+=i
			count+=1
			if(i>longest):
				longest=i

	average=0
	# print("In getRowAverage"+str(sum)+" "+str(count)+" L:"+str(longest))
	if count>0:
		average=sum/count

	return average

def getColumnAverage(image, threshold=250, averageThreshold=50):
	continuousPixel = []
	for i in range(image.shape[1]):
		continuousPixelCount=0
		for j in range(image.shape[0]):
			if image[j,i]>threshold:
				continuousPixelCount+=1
			else:
				continuousPixel.append(continuousPixelCount)
				continuousPixelCount=0
		continuousPixel.append(continuousPixelCount)

	sum=0
	count=0
	longest=-1
	for i in continuousPixel:
		if i > averageThreshold:
			sum+=i
			count+=1
			if(i>longest):
				longest=i

	average=0
	# print("In getColumnAverage: "+str(sum)+" "+str(count)+" Longest:"+str(longest))
	if count>0:
		average=sum/count

	return average


data_path = os.getcwd()+'/data'
dir_list = os.listdir(data_path)
	
car_count=0
bus_count=0
bike_count=0
im_dlist=[]
gray_scale_threshold=250
row_average_threshold=50
column_average_threshold=50

bus = []
car = []
bike = []


for dataset in dir_list:
	img_list=os.listdir(data_path+'/'+dataset)
	if dataset == 'BIKE':
		bike_count=len(img_list)
	if dataset == 'BUS':
		bis_count=len(img_list)
	if dataset == 'CAR':
		car_count=len(img_list)

	for img in img_list:
		print("Loading Image: "+str(dataset)+"/"+str(img))
		in_im=cv2.imread(data_path+'/'+dataset+'/'+img,cv2.IMREAD_GRAYSCALE)
		im_dlist.append([dataset,np.sum(in_im > gray_scale_threshold),
			getRowAverage(in_im,gray_scale_threshold,row_average_threshold),
			getColumnAverage(in_im,gray_scale_threshold,column_average_threshold)])
		if dataset == 'BIKE':
			bike.append(np.sum(in_im > 250))
		if dataset == 'BUS':
			bus.append(np.sum(in_im > 250))
		if dataset == 'CAR':
			car.append(np.sum(in_im > 250))
	print(dataset+"images loaded\n")
car.sort()
bus.sort()
bike.sort()
from random import shuffle
shuffle(im_dlist)

import matplotlib.pyplot as plt             
plt.plot(car)
plt.plot(bus)
plt.plot(bike)
#%%             
df=pd.DataFrame(im_dlist,columns=['Type','WCount','continuousPixelRowWise','continuousPixelColumnWise'])
#df = pd.DataFrame({'Type':im_dlist[:,0],
#	'WCount':im_dlist[:,1],
#	'continuousPixelRowWise':im_dlist[:,2],
#	'continuousPixelColumnWise':im_dlist[:,3]})
print(df)
plt.show()

df.to_csv("dataset.csv", sep=',', encoding='utf-8')