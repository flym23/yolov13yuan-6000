import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO

# ==================== 加载模型 ====================
model = YOLO('/home/rtx6000/ZZF/yolov13yuan-6000/ultralytics/cfg/models/v13/yolov13.yaml')
model.load('/home/rtx6000/ZZF/yolov13yuan-6000/yolov13n.pt')

# ==================== 开始训练 ====================
results = model.train(
    data='/home/rtx6000/ZZF/yolov13yuan-6000/data.yaml',
    epochs=200,
    patience=40,
    batch=16,          # A100 可以先从 16 试，显存够再到 32
    workers=8,         # Linux/A100 服务器不要用 0
    amp=False,          # 保持混合精度
    deterministic=True,  # 提速，但结果不再严格逐次复现
    plots=False,       # 减少训练过程绘图开销
    project='/home/rtx6000/ZZF/yolov13yuan-6000/runs/train',
    name='exp'
)
