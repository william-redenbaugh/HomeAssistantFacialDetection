from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from homeassistant_api import Client, State

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting webcam stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# setup our requests to the HASS server 
ip_addr = "192.168.1.53:8123"
api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2NDNkZWM0ZGVmNWM0M2QxOTIyMThmMDk3MjYwNzhmZCIsImlhdCI6MTY1NDMxODgyOCwiZXhwIjoxOTY5Njc4ODI4fQ._VV6Lgv4WPQVoZ1MUTyArs6KC305wUTAMoC2aobDwN8"
# Assigns the Client object to a variable and checks if it's running.
print("[INFO] starting up homeassistant client")
client = Client(ip_addr, api_token)
data = client.get_entities()
print(data)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	num_faces = 0
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		num_faces = num_faces + 1

	print("Num faces:", num_faces)
	#num_faces_entity = client.get_entity(entity_id='counter.living_room_face_count')
	#num_faces_entity.set_state(State(state = str(num_faces)))
	key = cv2.waitKey(1) & 0xFF
	time.sleep(1)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

vs.stop()