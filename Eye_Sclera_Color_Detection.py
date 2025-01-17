import cv2
import numpy as np
# import time

from numpy.ma.core import append



true_counter = 0
false_counter = 0

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

true_or_false = []

# Function to check for redness in the sclera
def is_sclera_red(eye_roi):
    hsv_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
    # Define the red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # Create masks for red regions
    mask1 = cv2.inRange(hsv_eye, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_eye, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Check if significant red region exists
    redness_ratio = cv2.countNonZero(mask) / (eye_roi.shape[0] * eye_roi.shape[1])
    return redness_ratio > 0.6  # What pct of th eye is red


# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam. Change this for an external camera.

if not cap.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        redness_flag = False
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Detect eyes within the face region
            face_roi = gray_frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_roi)

            for (ex, ey, ew, eh) in eyes:
                # Extract the eye region
                eye_roi = frame[y + ey:y + ey + eh, x + ex:x + ex + ew]

                if len(true_or_false) == 99:
                    for True_False in true_or_false:
                        if True_False == "True":
                            true_counter+=1
                            print ("True")
                        elif True_False == "False":
                            false_counter+=1
                            print("False")

                    if true_counter > 80:
                        cv2.putText(frame, "Redness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
                        print ("Redness Detected")
                    else:
                        cv2.putText(frame, "No Redness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
                        print("No Redness Detected")


                    true_or_false.clear()

                # Check for redness in the eye
                if is_sclera_red(eye_roi):
                    true_or_false.append("True")
                    redness_flag = True
                    # print("True")
                # Draw rectangle around the eyes
                else:
                    true_or_false.append("False")
                    # print('False')

                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(elapsed_time)

        # Show the frame with detection results
        cv2.imshow("Sclera Redness Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    # print(elapsed_time)
    # print()
    cap.release()
    cv2.destroyAllWindows()
