#OpenCV module for bbox search
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import random
import copy
#JUST WINDOWS  aka binary

#_-------Changed box drawing mode from vertical to with angle ------_#
class Bbox:
	def __init__(self):
		self.bboxInfo = {}
	def bubble_sort(self, items, numToReturn):
		""" Implementation of bubble sort """
		for i in range(len(items)):
			for j in range(len(items)-1-i):
					if items[j][1] < items[j+1][1]:
						items[j], items[j+1] = items[j+1], items[j] 
		return items[:numToReturn]
	def filterBboxes(self):
		for bboxObject in self.objects_to_bbox:
			if self.class_ratio[bboxObject]<self.bbox_filter:
				self.bboxInfo.pop(bboxObject)

		
	def drawLimitedObjects(self, segmented, raw, **kwargs):
		num_to_draw = kwargs['numToDraw']
		self.class_ratio = kwargs['classRatio']
		self.bbox_filter = kwargs['bboxFilter']
		self.objects_to_bbox = kwargs['classesToBbox']
		listToDraw = []
		self.filterBboxes()
		keys = self.bboxInfo.keys()

		for key in keys:
			for x in range(self.bboxInfo[key].__len__()):
				area = self.bboxInfo[key][x][4]
				listToDraw.append([key, area, self.bboxInfo[key][x]])
		listToDraw = self.bubble_sort(listToDraw, num_to_draw)
		print listToDraw
		segmentedImageRGB = np.array(segmented)
		output_im = np.array(raw)

		JSONCoords = {}
		if num_to_draw>listToDraw.__len__():
			num_to_draw = listToDraw.__len__()

		for i in range(num_to_draw):
			x1 = listToDraw[i][2][0]
			y1 = listToDraw[i][2][1]
			x2 = listToDraw[i][2][0] + listToDraw[i][2][2]
			y2 = listToDraw[i][2][1] + listToDraw[i][2][3]
			clr = listToDraw[i][2][5]
			box = cv2.cv.BoxPoints(((listToDraw[i][2][0],listToDraw[i][2][1]),(listToDraw[i][2][2],listToDraw[i][2][3]),listToDraw[i][2][6])) # cv2.boxPoints(rect) for OpenCV 3.x
			box = np.int0(box)
			cv2.drawContours(segmentedImageRGB,[box],0,clr,2)
			cv2.drawContours(output_im,[box],0,clr,2)

		BboxedRawImage = Image.fromarray(output_im)
		for x in range(num_to_draw):
			listToDraw[x][2].pop()
			listToDraw[x][2].pop()
			if JSONCoords.has_key(listToDraw[x][0]):
				JSONCoords[listToDraw[x][0]].append(listToDraw[x][2])
			else:
				JSONCoords[listToDraw[x][0]]=[listToDraw[x][2]]
		BboxedImage = Image.fromarray(segmentedImageRGB)
		return BboxedImage, BboxedRawImage, JSONCoords #image
		




	def findBbox(self, segmentedImageRGB, maskImage,  **kwargs):
		objectName = kwargs['objectName']

		maskImage = maskImage.convert("RGB")
		imW,imH = segmentedImageRGB.size
		open_cv_image = np.array(maskImage) 
		open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

		imgray = cv2.cvtColor(open_cv_image,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,127,255,0)
		contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		self.bboxInfo[objectName] = [] #[[coords,clr],[coords,clr]]
		bboxClr = (int(math.floor(random.random()*255)), int(math.floor(random.random()*255)), int(math.floor(random.random()*255)))
		for c in contours:
			rect = cv2.minAreaRect(c)
			if (rect[1][0]*rect[1][1])<500: continue  
			x = int(rect[0][0])
			y = int(rect[0][1])
			w = int(rect[1][0])
			h = int(rect[1][1])
			rotation = rect[2]
			single_bbox_info = [x,y,w,h,w*h,bboxClr, rotation]
			#Instrument to filter inner bboxes
			if thresh[y][x]==0: continue
			self.bboxInfo[objectName].append(single_bbox_info)
