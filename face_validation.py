# libraries
import os
import cv2 as cv
import numpy as np
from collections import Counter

# load utilities
classifier_path = "utils/haarcascade_frontalface_alt2.xml"

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


def compare_histogram(a_hist, b_hist, method=cv.HISTCMP_CORREL):
	if isinstance(a_hist, list) and isinstance(b_hist, list):
		diff = []
		for i, channel in enumerate(CHANNELS):
			diff += [cv.compareHist(a_hist[i], b_hist[i], method=method)]
	else:
		diff = cv.compareHist(a_hist, b_hist, method=method)

	return np.mean(diff)

def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')

# define blank list for cropped image
cropped_face = []

if __name__ == "__main__":
	# define detector
	detector = cv.CascadeClassifier(classifier_path)
	# input image
	filename = "utils/data/1.jpeg"
	# apply coordinates
	faces = get_coordinate(filename)
	# read image file in BGR -> RGB
	img = cv.imread(filename)
	# bounding box & crop image
	for face in faces:
		x, y, w, h = [x for x in face]
		# push crop face to blank list
		cropped_face.append(cv.resize(img[y:y+h, x:x+w],
		                              (200, 200),
		                              interpolation = cv.INTER_AREA))

	# show croppped image
	# for face in cropped_face:
	# 	cv.imshow('face', face)
	# 	cv.waitKey(0)

	for x, face1 in enumerate(cropped_face):
		for y, face2 in enumerate(cropped_face):
			print(x, y)
			# convert image into histogram
			h1, h2 = np.histogramdd(face1.flatten()), \
			         np.histogramdd(face2.flatten())
			# calculate histogram pixel using euclidean
			result = compare_histogram(np_hist_to_cv(h1),
			                           np_hist_to_cv(h2))
			print(result)
