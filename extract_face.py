# load libraries
import os
import cv2 as cv

# load utilities
classifier_path = "utils/haarcascade_frontalface_alt2.xml"
folder = 'utils/data/'

# coordinate face
def get_coordinate(filename):
	# load image -> BGR to RGB to grayscale
	img = cv.cvtColor(cv.cvtColor(cv.imread(filename),
	                              cv.COLOR_BGR2RGB),
	                   cv.COLOR_RGB2GRAY)
	# face detection using haarcascade classifiers
	detections = detector.detectMultiScale(image=img)
	# convert list result into tuple
	faces = sorted(tuple(data) for data in list(detections))
	# output
	return tuple(faces)

if __name__ == "__main__":
	for idx, file in enumerate(os.listdir(folder)):
		# define detector
		detector = cv.CascadeClassifier(classifier_path)
		# input image
		filename = folder + file
		# apply coordinates
		faces = get_coordinate(filename)
		# read image file in BGR -> RGB
		img = cv.imread(filename)
		# bounding box & crop image
		for idx_y, face in enumerate(faces):
			x, y, w, h = [x for x in face]
			# push crop face to blank list
			cv.imwrite(f"utils/data_1/{idx}_{idx_y}.jpg", cv.resize(img[y:y + h, x:x + w],
			                              (200, 200),
			                              interpolation=cv.INTER_AREA))
		print(f"[INFO] - {idx} is done")
