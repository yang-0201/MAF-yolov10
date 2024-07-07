from ultralytics import YOLOv10

model = YOLOv10('runs/detect/MAFMS-YOLO-ATSS11/weights/best.pt')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()