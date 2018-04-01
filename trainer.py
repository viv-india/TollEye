#importing libraries
import os #to get working directory
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from sklearn import preprocessing
import matplotlib.pyplot as pyplot
from keras import callbacks
from keras.models import model_from_json
from keras.models import load_model

global data_path #path to data set
global dir_list #contains dirctories in main data set file
global classes #number of classes to recognize
global img_r #rows in image
global img_c #columns in image
global ch #number of channels 
global epoch #number of epochs
global truck_count
global mini_truck_count
global bus_count
global car_count
global X_test
global Y_test

	
############# DATA LOADING ##################
def get_data():
	data_path = os.getcwd()+'/vehicles'
	dir_list = os.listdir(data_path)
	classes = 4 #car,bus,mini-truck,truck
	img_r=128
	img_c=128
	ch=1
	epoch=15
########### DATA FORMATTING #################
def format_data():
	for dataset in dir_list:
		img_list=os.listdir(data_path+'/'+dataset)
	
	if dataset == 'CAR':
		car_count=img_list.shape[0]
	if dataset == 'TRUCK':
		truck_count=img_list.shape[0]
	if dataset == 'BUS':
		bus_count=img_list.shape[0]
	if dataset == 'MINI_TRUCK':
		mini_truck_count=img_list.shape[0]
	

	img_dlist=[] #image array
	print(dataset+"images loaded\n")
	for img in img_list:
		in_im=cv2.imread(data_path+'/'+dataset+'/'+img)
		in_im=cv2.cvtColor(in_im,cv2.COLOR_BGR2GRAY) #GRAY CONVERSION
		in_im=cv2.resize(in_im,(128,128)) #resize to make all images of same size
		image_dlist.append(in_im)

	data=np.array(im_dlist)
	data=data.astype('float32')
	data /=255 #making all values from 0 to 1 to make training converge quick
	print(data.shape)
	data=np.rollaxis(data,3,1)
	print(data.shape)
	return data

def data_labeling(data):
	sample_count=data.shape[0]
	labels = np.zeroes((sample_count),dtype='int64')
	total=0
	labels[total : total+car_count] =0;
	total=car_count
	labels[total : total+ bus_count] =1;
	total+= bus_count
	labels[total : total+ truck_count] =2;
	total+= truck_count
	labels[total : total+ mini_truck_count] =3;
	total+= mini_truck_count

	label_names = ['CAR','BUS','TRUCK','MINI-TRUCK']
	Y = np_utils.to_categorical(labels,classes)#one hot encoding
	x,y = shuffle(img_data,Y,random_state=2)#shuffling data to make system strong 
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2) #giving 20% data for testing

def model_defination(data):
	in_shape=data[0].shape

	model=Sequential()
	model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=in_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

	# Viewing model_configuration

	model.summary()
	model.get_config()
	model.layers[0].get_config()
	model.layers[0].input_shape			
	model.layers[0].output_shape			
	model.layers[0].get_weights()
	np.shape(model.layers[0].get_weights()[0])
	model.layers[0].trainable
	return model
def train_model(model):	
	hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
	filename='model_train_new.csv'
	csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
	early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
	filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
	checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [csv_log,early_stopping,checkpoint]
	hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)

	# visualizing losses and accuracy
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(num_epoch)
	plt.figure(1,figsize=(7,5))
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.figure(2,figsize=(7,5))
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'],loc=4)
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.show()
def model_evaluation(model):
	score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
	print('Test Loss:', score[0])
	print('Test accuracy:', score[1])
	test_image = X_test[0:1]
	print (test_image.shape)
	print(model.predict(test_image))
	print(model.predict_classes(test_image))
	print(y_test[0:1])
	
def save_model(model):
	# serialize model to json
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
  	  json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

