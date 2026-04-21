from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("E:/PPE-Kit-Detection-System/runs/detect/train/weights/best.pt")

    model.val(data = "dataset/data.yaml")
