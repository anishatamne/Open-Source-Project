from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=384,
        batch=16,
        device=0,
        pretrained = True
    )

if __name__ == "__main__":
    main()