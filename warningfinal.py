# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:59:53 2020

@author: Sarath
"""


#python CrowdDetection.py --prototxt Objects.prototxt.txt --model Objects.caffemodel


#importing all the packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime
import matplotlib.pyplot as plt
import csv
import threading
import pyttsx3
import keras
import pickle

attt = 0
stop_thread = False

def detector():
	global attt, stop_thread
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-p", "--prototxt", required=True,
	#	help="path to Caffe 'deploy' prototxt file")
	#ap.add_argument("-m", "--model", required=True,
	#	help="path to Caffe pre-trained model")
	#ap.add_argument("-c", "--confidence", type=float, default=0.2,
	#	help="minimum probability to filter weak detections")
	#args = vars(ap.parse_args())

	CLASSES = ["", "", "", "", "",
		"", "", "", "", "", "", "",
		"", "", "", "person", "", "",
		"", "", ""]
	color = np.random.uniform(200,200,200)
	threshold = 0.999        # PROBABLITY THRESHOLD
	font = cv2.FONT_HERSHEY_SIMPLEX
	pickle_in=open("S:\openlab\model_trained.p","rb")  ## rb = READ BYTE
	model=pickle.load(pickle_in)
	 
	def grayscale(img):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img
	def equalize(img):
		img =cv2.equalizeHist(img)
		return img
	def preprocessing(img):
		img = grayscale(img)
		img = equalize(img)
		img = img/255
		return img
	def getCalssName(classNo):
		if   classNo == 0: return 'Speed Limit 20 km/h'
		elif classNo == 1: return 'Speed Limit 30 km/h'
		elif classNo == 2: return 'Speed Limit 50 km/h'
		elif classNo == 3: return 'Speed Limit 60 km/h'
		elif classNo == 4: return 'Speed Limit 70 km/h'
		elif classNo == 5: return 'Speed Limit 80 km/h'
		elif classNo == 6: return 'End of Speed Limit 80 km/h'
		elif classNo == 7: return 'Speed Limit 100 km/h'
		elif classNo == 8: return 'Speed Limit 120 km/h'
		elif classNo == 9: return 'No passing'
		elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
		elif classNo == 11: return 'Right-of-way at the next intersection'
		elif classNo == 12: return 'Priority road'
		elif classNo == 13: return 'Yield'
		elif classNo == 14: return 'Stop'
		elif classNo == 15: return 'No vechiles'
		elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
		elif classNo == 17: return 'No entry'
		elif classNo == 18: return 'General caution'
		elif classNo == 19: return 'Dangerous curve to the left'
		elif classNo == 20: return 'Dangerous curve to the right'
		elif classNo == 21: return 'Double curve'
		elif classNo == 22: return 'Bumpy road'
		elif classNo == 23: return 'Slippery road'
		elif classNo == 24: return 'Road narrows on the right'
		elif classNo == 25: return 'Road work'
		elif classNo == 26: return 'Traffic signals'
		elif classNo == 27: return 'Pedestrians'
		elif classNo == 28: return 'Children crossing'
		elif classNo == 29: return 'Bicycles crossing'
		elif classNo == 30: return 'Beware of ice/snow'
		elif classNo == 31: return 'Wild animals crossing'
		elif classNo == 32: return 'End of all speed and passing limits'
		elif classNo == 33: return 'Turn right ahead'
		elif classNo == 34: return 'Turn left ahead'
		elif classNo == 35: return 'Ahead only'
		elif classNo == 36: return 'Go straight or right'
		elif classNo == 37: return 'Go straight or left'
		elif classNo == 38: return 'Keep right'
		elif classNo == 39: return 'Keep left'
		elif classNo == 40: return 'Roundabout mandatory'
		elif classNo == 41: return 'End of no passing'
		elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe("Objects.prototxt.txt", "Objects.caffemodel")
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()


	personId = []
	f = open("data.csv","w+",newline='')
	# attt=0
	t = time.strftime("%I:%M:%S")
	t.strip("2019-12-11")
	pltime=str(t)
	pltime=pltime.split(':')
	mm=int(pltime[1])
	timecheck=mm

	 


	while True:
		t = time.strftime("%I:%M:%S")
		t.strip("2019-12-11")

		pltime=str(t)
		pltime=pltime.split(':')
		hh=int(pltime[0])
		mm=int(pltime[1])

		pid = 0
		personId.clear()

		frame = vs.read()
		
		frame = imutils.resize(frame, width=400)

		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		
		# predictions:
		net.setInput(blob)
		detections = net.forward()
		img = np.asarray(frame)
		img = cv2.resize(img, (32, 32))
		img = preprocessing(img)
		#cv2.imshow("Processed Image", img)
		img = img.reshape(1, 32, 32, 1)
		cv2.putText(frame, "CLASS: " , (20, 35), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
		cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
		# PREDICT IMAGE
		predictions = model.predict(img)
		classIndex = model.predict_classes(img)
		probabilityValue =np.amax(predictions)
		if probabilityValue > threshold:
			#print(getCalssName(classIndex))
			cv2.putText(frame,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.putText(frame, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			sign = str(getCalssName(classIndex))
			def speechhehe():
				engine = pyttsx3.init() 
				engine.say(sign)
				engine.runAndWait()
							
			ssxthehe = threading.Thread(target=speechhehe)
			ssxthehe.start()
			
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.2:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				#labeling the pesron with id
				label = "id = {} {}: {:.2f}%".format(pid, CLASSES[idx],
					confidence * 100)
				pid += 1
				personId.append(pid)
				personId.sort()
				cv2.rectangle(frame, (startX, startY), (endX-100, endY),
					color[idx], 2)
				centerX = (startX+endX)//2
				centerY = (startY+endY)//2
				coord = (centerX, centerY)

				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[idx], 2)
					#To FInd The Total
				try:
					attt+=personId[-1]
					def speech():
						engine = pyttsx3.init() 
						if(attt>4):
							engine.say("Crowdy area")
							engine.runAndWait()
							
					ssxt_ = threading.Thread(target=speech)
					ssxt_.start()
				except IndexError:
					pass
				# file writing
				if True:
					try:
						
						writer = csv.writer(f)
						writer.writerow([str(hh)+":"+str(mm), int(attt)])
						attt=0
						timecheck=mm
						
					except IndexError:
						personId.clear()
						personId.append(0)


		# show the output frame
		cv2.imshow("Frame", frame)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			print("Final pid: "+str(personId))
			break

		# update the FPS counter
		fps.update()



	#stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	stop_thread = True

	cv2.destroyAllWindows()
	vs.stop()




detect_ = threading.Thread(target=detector)
detect_.start()

