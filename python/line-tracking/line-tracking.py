import cv2
import numpy as np

# --- Hardcoded Parameters ---
# We will filter for YELLOW
# In OpenCV, H: 0-179, S: 0-255, V: 0-255
LOWER_COLOR = np.array([20, 100, 100])
UPPER_COLOR = np.array([30, 255, 255])

# --- Image Path ---
image_path = "testline.jpg"  # <--- CHANGE THIS to the path of your image

# --- 1. Load Image ---
frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Cannot load image from {image_path}")
    print("Please make sure the file path is correct.")
    exit()

# Let's resize to a standard size (e.g., 640x480)
frame = cv2.resize(frame, (640, 480))
h, w, _ = frame.shape
print(f"Image size: {w}x{h}")

# --- 2. OpenCV Processing (HSV Filter) ---
    
# Convert to HSV
mat_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
# Create mask using the selected color
mat_mask = cv2.inRange(mat_hsv, LOWER_COLOR, UPPER_COLOR)
    
# --- 3. Show Results ---
print("Showing Original Image and HSV Mask. Press 'q' to quit.")

# Show the mask in a separate window
# This now shows the HSV result for the FULL IMAGE
cv2.imshow("Mask (HSV Result)", mat_mask)

# Show the main frame
cv2.imshow("Original Image", frame)

# Wait for a key press and then exit
cv2.waitKey(0)

# --- Cleanup ---
cv2.destroyAllWindows()