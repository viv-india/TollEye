from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')
import cv2,os
def load_model()
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	 model = model_from_json(loaded_model_json)
	# load weights into model
	model.load_weights("model.h5")
	print("Loaded model from disk")
	model=load_model('model.hdf5')

def test_image(model,test_img):
	test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	test_image=cv2.resize(test_image,(128,128))
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	test_image= np.expand_dims(test_image, axis=0)
	test_image= np.expand_dims(test_image, axis=0)
	# Predicting the test image
	print((model.predict(test_image)))
	print(model.predict_classes(test_image))

def start_video():
	vid_path=os.getcwd()+'/'+'test_video.mp4'
	vidcap = cv2.VideoCapture(vid_path)
	success,image = vidcap.read()
	x_car=300 #600
	y=200
	x_other=100 #truck,bus,mini-truck
	count = 0
	print(success)
	while success:
	      success,image = vidcap.read()
	      image1=image[x_truck:x_truck+600, y:y+600]
	      image2=image[x_truck:x_car+600, y:y+600]
	      test_image(image1)
	      test_image(image2)
load_model()
start_video()
