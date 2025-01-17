import cv2
import cvzone
import numpy as np

# Load Haar cascades for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


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
    return redness_ratio > 0.6  # What pct of the eye is red



# List of image paths to process
image_paths = ["Test_Image_2.jpg"]

for i in range(617):
    image_paths.append(str((str(i+2) + ".jpg")))
print(image_paths)
# Process each image
for image_path in image_paths:
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        continue

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    redness_found = False  # Initialize a flag for redness detection

    for (x, y, w, h) in eyes:
        # Extract the eye region
        eye_roi = frame[y:y + h, x:x + w]
        # print(eye_roi)
        # cv2.circle(frame, (eye_roi[0], eye_roi[1]), 10, (0, 0, 255), -1)

        # Check for redness in the eye
        if is_sclera_red(eye_roi):
            redness_found = True  # Set flag if redness is found
            # cvzone.putTextRect(frame, "Redness Detected", (50, 100), colorR=(255, 0, 255))
            break  # Exit loop if one eye is detected with redness

    if redness_found:
        print(image_path + "Redness detected!")  # Print once if any eye is red
    else:
        print(image_path + "No Redness Detected!")  # Print once if no eyes are red

    # Draw rectangles around the detected eyes regardless of the redness
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with detection results
    # cv2.imshow("Sclera Redness Detection", frame)
    cv2.waitKey(0)  # Wait for a key press to close the image window

# Close all OpenCV windows
cv2.destroyAllWindows()
