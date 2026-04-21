from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n-pose.pt")

# Read image
img = cv2.imread("image.jpg")

# Check if image loaded correctly
if img is None:
    raise ValueError("Image not found or failed to load.")

# Run inference
results = model(img)

# Show output (only if results exist)
if results and len(results) > 0:
    annotated_frame = results[0].plot()
    cv2.imshow("Pose", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No results returned from model")

# Access keypoints safely
if results and len(results) > 0:
    result = results[0]

    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints = result.keypoints.xy
        print("Keypoints detected:")
        print(keypoints)
    else:
        print("No keypoints detected")
else:
    print("Skipping keypoint extraction due to no detections")