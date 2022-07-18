import imutils
from imutils.video import VideoStream
from imutils import face_utils
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import dlib


# Uzaklik fonksiyonu
def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)


# EAR hesabi fonksiyonu
def eye_aspect_ratio(eye):
	# Goz kapaklari arasindaki dikey uzaklik
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	# Gozun yatay uzunlugu
	C = euclidean_dist(eye[0], eye[3])
	# Eye Aspect Ratio hesabi
	ear = (A + B) / (2.0 * C)
	return ear


# MAR hesabi fonksiyonu
def mouth_aspect_ratio(mouth):
	M1 = euclidean_dist(mouth[2], mouth[10])
	M2 = euclidean_dist(mouth[4], mouth[8])
	M3 = euclidean_dist(mouth[0], mouth[6])
	mar = (M1 + M2) / (2.0 * M3)
	return mar


#Conterlari sifirlama
counter = 0
Mcounter = 0
Scounter = 0
Ycounter = 0
mtimecounter = 0

# HAAR Cascade landmark predictor yuklenmesi
print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream...")

vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:

		rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

		# Facial landmarklari numpy dizisine donusturme
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Sag ve sol goz icin EAR hesaplama
		leftEye = shape[42:48]
		rightEye = shape[36:42]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# agiz icin MAR hesaplama
		mouthpoints = shape[48:68]
		mouthMAR = mouth_aspect_ratio(mouthpoints)

		# Iki gozun EAR degerlerinin ortalamasini alma
		ear = (leftEAR + rightEAR) / 2.0

		# Gozleri isaretleme
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)

		#Agizi isaretleme
		mouthHull = cv2.convexHull(mouthpoints)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

		if ear < 0.25:
			counter = counter + 1

			if counter >= 15:
				cv2.putText(frame, "DROWSINESS ALERT!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				Scounter = Scounter + 1

		else:
			counter = 0

		cv2.putText(frame, "EAR: {:.3f}".format(ear), (5, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.3f}".format(mouthMAR), (5,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
		cv2.putText(frame, "Yawn Counter: {:.0f}".format(Ycounter), (250, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

		if mouthMAR > 0.75:
			mtimecounter  = mtimecounter + 1

			if mtimecounter >= 10:
				cv2.putText(frame, "YAWN DETECTED!!!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				ywn = True

				if ywn == True:
					Ycounter = Ycounter + 1
					ywn = False

		else:
			mtimecounter = 0

	cv2.imshow("FRAME", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):

		break

cv2.destroyAllWindows()
vs.stop()
