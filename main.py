import os
from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == "__main__":
    rf = Roboflow(api_key="f2Zm3PDjLvH2GrAMKRuE")
    project = rf.workspace("test-p4ko4").project("test2-xzapr")
    version = project.version(7)
    dataset = version.download("yolov9")

    model = YOLO("yolov9t.pt")
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=1000,
        imgsz=640,
    )
