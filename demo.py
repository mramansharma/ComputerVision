from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import sin,cos

yellow = (255,255,0)
 
def show(image):
	#figure size in inches
	plt.figure(figsize=(10,10))
	plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
	#mask the mask rgb
	rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
	img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
	return img

def find_lemons_set(image):
	#copy image
	image = image.copy()
	im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#isolating the largest contour
	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	biggest_contour = max(contour_sizes, key=lambda x: x[0])[1] 

	#return the biggest contour
	mask = np.zeros(image.shape, np.uint8)
	cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
	return biggest_contour, mask

def circle_contour(image, contour):
	#bounding ellipse
	image_with_ellipse = image.copy()
	ellipse = cv2.fitEllipse(contour)
	#add it
	cv2.ellipse(image_with_ellipse, yellow, 2, cv2.LINE_AA)
	return image_with_ellipse

def find_lemons(image):
	#*: Convert the correct color scheme
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#*: Scale our image properly
	max_dimension = max(image.shape)
	scale = 930/max_dimension
	image = cv2.resize(image, None, fx=scale, fy=scale)

	#*: Clean our image
	image_blur = cv2.GaussianBlur(image, (7,7), 0)
	image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

	#*: Define filters
	min_yellow = np.array([100,100, 0])
	max_yellow = np.array([255,255, 0])

	mask1 = cv2.inRange(image_blur_hsv, min_yellow, max_yellow)

	#*: Brightness
	min_yellow2 = np.array([170,100, 0])
	max_yellow2 = np.array([180,256,0])

	mask2 = cv2.inRange(image_blur_hsv, min_yellow2, max_yellow2)


	#*: Combine mask
	mask = mask1+mask2

	#*: Segementation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
	mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

	#*: Find lemonsFound
	big_lemons_contour, mask_lemons = find_lemons_set(mask_clean)

	#*: overlay the mask that we created on the image
	overlay = overlay_mask(mask_clean, image)

	#*: circle the biggest lemons
	circled = circle_contour(overlay, big_lemons_contour)

	show(circled)

	#*: Convert back to the original color scheme
	bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
	return bgr

#*: read the image
image = cv2.imread('lemons.jpg')
result = find_lemons(image)
#: write the new image
cv2.imwrite('lemonsFound.jpg', result)
