import cv2
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
