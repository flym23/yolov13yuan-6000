from ultralytics import YOLO

# 加载断点，自动续训
model = YOLO("/home/zhengzf/.virtualenvs/yolov13/runs/train/exp/weights/last.pt")
model.train(resume=True)