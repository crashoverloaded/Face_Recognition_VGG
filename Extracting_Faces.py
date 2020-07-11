import os 
from os import listdir
from os.path import isfile, join
import cv2



def makedir(directory):
		
	if not os.path.exists(directory):
		os.makedirs(directory)
		return None
	
	else:
		pass

face_detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

my_path = "./people/"
image_file_names = [i for  i in listdir(my_path) if isfile(join(my_path,i))]
makedir("./group_of_faces/")


for image in image_file_names:
	person_image = cv2.imread(my_path+image)
	face = face_detector.detectMultiScale(person_image , 1.3  ,5)
	if face is not None:
		for (x,y,w,h) in face:
		
			faces = person_image[y:y+h , x:x+w]
			roi = cv2.resize(faces , (128 ,128) , interpolation = cv2.INTER_CUBIC)
		
	path = "./group_of_faces/" + "face_"+image
	try:
		cv2.imwrite(path , roi)
		cv2.imshow("face" , roi)
		cv2.waitKey(0)
	except NameError:
		print('Failed to set "ROI"')
cv2.destroyAllWindows()
