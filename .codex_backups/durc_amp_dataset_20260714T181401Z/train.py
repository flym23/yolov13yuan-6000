import os

# ==================== 环境配置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第2块GPU
os.environ["WANDB_DISABLED"] = "true"  # 禁用wandb日志

# ==================== 导入依赖 ====================
from ultralytics import YOLO


# ==================== 模型加载 ====================
# 加载YOLOv13配置文件和预训练权重
model = YOLO(
    "/home/rom305/zzf/yolov13-305-yuan/ultralytics/cfg/models/v13/yolov13.yaml"
)
model.load("/home/rom305/zzf/yolov13-305-yuan/yolov13n.pt")


# ==================== 开始训练 ====================
if __name__ == "__main__":
    results = model.train(
        data="/home/rom305/zzf/yolov13-305-yuan/data.yaml",  # 数据集配置
        epochs=200,
        patience=40,
        batch=16,  # A100 可以先从 16 试，显存够再到 32
        workers=8,  # Linux/A100 服务器不要用 0
        amp=False,  # 统一禁用 AMP，使用 FP32 训练
        deterministic=False,
        plots=False,  # 减少训练过程绘图开销
        project="/home/rom305/zzf/yolov13-305-yuan/runs/train",  # 结果保存路径
        name="exp",  # 实验名称
    )
