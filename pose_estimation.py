from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n-pose.pt")

# Read image
img = cv2.imread("image.jpg")

# Set confidence threshold
CONF_THRESHOLD = 0.5  # 🔧 You can tune this (0.3–0.7 recommended)

# Run inference with confidence filtering
results = model(img, conf=CONF_THRESHOLD)

# Show output
annotated_frame = results[0].plot()
cv2.imshow("Pose", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Access keypoints
keypoints = results[0].keypoints.xy
print(keypoints)