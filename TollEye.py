from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')
import cv2,os
import numpy as np
from tkinter import*
from tkinter import filedialog
import tkinter.filedialog 
global x
global d
import csv
import pandas as pd

################################### PREDICTION MODEL #############################################
model=load_model('model.hdf5')
label_names = ['BIKE', 'CAR', 'TRUCK', 'BIKE', 'MINITRUCK']


def test_image(test_image):
	
	test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	test_image=cv2.resize(test_image,(128,128))
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	num_channel=1


	if num_channel==1:
		if K.image_dim_ordering()=='th':
			test_image= np.expand_dims(test_image, axis=0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=3) 
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
			
	else:
		if K.image_dim_ordering()=='th':
			test_image=np.rollaxis(test_image,2,0)
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)
		else:
			test_image= np.expand_dims(test_image, axis=0)
			print (test_image.shape)

	return label_names[model.predict_classes(test_image)[0]]
	
################################################################################################

def askd():	
	global filename
	global im_set
	global j
	j=0
	filename = filedialog.askdirectory()
	im_set=os.listdir(filename)
	

def img_resize(im):
	test_image=cv2.imread(im)
	test_image=cv2.resize(test_image,(500,500))
	cv2.imwrite('temp.png',test_image)

def Click():
	global j
	j=j+1
	img_resize(filename+'/'+im_set[j]) 	
	img=PhotoImage(file='temp.png')
	image=cv2.imread(filename+'/'+im_set[j])
	Detected_type=test_image(image)
	canvas.create_image(30,30,anchor=NW,image=img)
	canvas.delete('all')
	canvas.create_image(30,30,anchor=NW,image=img)
	
	canvas.create_text(700,100,fill="Blue",font="Times 30 bold",text=Detected_type)
	
	d[Detected_type]=d[Detected_type]+1;
	canvas.create_text(700,200,fill="Blue",font="Times 10 bold",text="CAR"+' = '+str(d["CAR"]) )
	canvas.create_text(700,230,fill="Blue",font="Times 10 bold",text="BIKE"+' = '+str(d["BIKE"]) )
	canvas.create_text(700,260,fill="Blue",font="Times 10 bold",text="TRUCK"+' = '+str(d["TRUCK"]))
	canvas.create_text(700,290,fill="Blue",font="Times 10 bold",text="MINI-TRUCK"+' = '+str(d["MINITRUCK"]) )
	
	with open('data.csv', 'w') as csvfile:   
		fieldnames = ['TYPE', 'COUNT']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow({'TYPE': 'CAR', 'COUNT': d["CAR"]})
		writer.writerow({'TYPE': 'TRUCK', 'COUNT': d["TRUCK"]})
		writer.writerow({'TYPE': 'BIKE', 'COUNT': d["BIKE"]})
		writer.writerow({'TYPE': 'MINITRUCK', 'COUNT': d["MINITRUCK"]})
		
	mainloop()

global j
j=-1
root =Tk()
global d
d={}
d["CAR"]=0
d["BIKE"]=0
d["TRUCK"]=0;
d["MINITRUCK"]=0;
labelframe = LabelFrame(root)
labelframe.pack(fill="both", expand="yes")
left = Label(labelframe)
button1=Button(labelframe, padx = 0, pady = 0, text="Connect",command = askd)
button1.pack(side=LEFT)

button=Button(labelframe, padx = 5, pady = 5, text="Next",command = Click)
button.pack(side = RIGHT)



canvas= Canvas(root,width=1000,height=500)
canvas.pack(expand=YES,fill=BOTH)
canvas.create_text(500,250,fill="Blue",font="Times 50 bold",text="TOLL EYE")
canvas.create_text(510,300,fill="Blue",font="Times 10 bold",text="*Click on connect to add the image source")
canvas.create_text(510,310,fill="Blue",font="Times 10 bold",text="*Click on next button to Start")
x=1

mainloop()

