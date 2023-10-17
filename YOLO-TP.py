from ultralytics import YOLO
import os

# import torch

# device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
# Load a model
model = YOLO("yolov8n-cls.pt")  # build a new model from scratch

# Use the model
# results = model.train(data="data.yaml", epochs=1)  # train the model

# results = model.train(data="data.yaml", epochs=100, imgsz=64)
datadir = os.getcwd() + "/dataset"
results = model.train(data=datadir, epochs=20, imgsz=224)

# yolo task=classify mode=train model=yolov8n-cls.pt data=/Users/ibraheemassiri/Desktop/ey/dataset epochs=10 imgsz=192
# imgsz=[196] must be multiple of max stride 32, updating to [224]
