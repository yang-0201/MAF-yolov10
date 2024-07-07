from ultralytics import YOLOv10
import torch
if __name__ == '__main__':
    model = YOLOv10('MAFMS-yolov10n-up-aux.yaml')

    model.train(data='coco.yaml'  ,epochs=500 ,batch=80, imgsz=640 ,device=[0,1,2,3] ,val_period=5)
