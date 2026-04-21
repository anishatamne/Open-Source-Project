from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n-pose.pt")

# Read image
img = cv2.imread("image.jpg")

# Validate input
if img is None:
    raise ValueError("Image not found or failed to load. Check file path.")

# Run inference
results = model(img)

# Show output
annotated_frame = results[0].plot()
cv2.imshow("Pose", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Access keypoints
keypoints = results[0].keypoints.xy
print(keypoints)
