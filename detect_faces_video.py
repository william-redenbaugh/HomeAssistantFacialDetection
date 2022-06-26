from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import requests
import threading

num_faces = 0
num_faces_lock = threading.Lock()

def hass_integration_thread(name):
	global num_faces, num_faces_lock
	# Assigns the Client object to a variable and checks if it's running.
	print("[INFO] starting up homeassistant client")

	# setup our requests to the HASS server 
	base_url = 'http://192.168.1.53:8123/api/states/counter.living_room_face_count'
	api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2NDNkZWM0ZGVmNWM0M2QxOTIyMThmMDk3MjYwNzhmZCIsImlhdCI6MTY1NDMxODgyOCwiZXhwIjoxOTY5Njc4ODI4fQ._VV6Lgv4WPQVoZ1MUTyArs6KC305wUTAMoC2aobDwN8"
	headers = {'Authorization': f'Bearer {api_token}'}
	req = {
		"state": "0",
		"attributes": {
			"num_faces":"1"
		}}
	response = requests.post(f'{base_url}', headers=headers, json=req)

	while True: 
		num_f = 0
		with num_faces_lock:
			num_f = num_faces

		req = {
		"state": str(num_f),
		"attributes": {
			"num_faces":str(num_f)
		}}
		response = requests.post(f'{base_url}', headers=headers, json=req)	
		time.sleep(2)

def face_detect_thread():
	global num_faces, num_faces_lock

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] starting webcam stream...")
	vs = VideoStream(src=0).start()

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
	
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
	
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()
		
		num_f = 0
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < 0.5:
				continue
			num_f = num_f + 1

		with num_faces_lock:
			num_faces = num_f

		print(num_f)
		time.sleep(1)

	vs.stop()

hass_integration_thread_handler = threading.Thread(target=hass_integration_thread, args=(1,))
hass_integration_thread_handler.start()
face_detect_thread()
