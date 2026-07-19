#!/usr/bin/env python3
"""
使用 ultralytics YOLO 对指定文件夹的全部图片进行推理，
为不同类别使用不同颜色的目标框并保存到指定目录。

权重：/home/zhengzf/.virtualenvs/yolov13/runs/train/exp12-声呐-1/weights/best.pt
输入：/home/zhengzf/.virtualenvs/yolov5-master/datasets/images/test
输出：/home/zhengzf/.virtualenvs/yolov13/runs/pred
"""
import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _color_for_class(name: str) -> tuple:
    # 固定常见类别颜色（BGR），其余类别采用可复现哈希色
    fixed = {
        # 你提供的目标类别颜色（BGR）
        'frame': (0, 255, 255),        # 黄色
        'cage': (255, 192, 203),       # 粉色
        'rov': (165, 42, 42),          # 棕色
        'fish': (0, 0, 255),           # 红色
        'hook': (200, 200, 0),
        'anchor': (128, 0, 128),       # 紫色
        'tire': (0, 255, 0),           # 绿色
        'plastic bucket': (255, 0, 0), # 蓝色
        'Oil drums': (255, 255, 0),    # 青色
    }
    if name in fixed:
        return fixed[name]
    h = abs(hash(name)) % 360
    # 将H映射到BGR（简易HSV→BGR近似），保证颜色区分度
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _auto_increment_dir(project: Path, name: str) -> Path:
    base = project / name
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    i = 2
    while True:
        cand = project / f"{name}{i}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        i += 1


def run_folder_infer():
    weights = '/home/zhengzf/.virtualenvs/yolov13/runs/train/exp2/weights/best.pt'
    source_dir = Path('/home/zhengzf/.virtualenvs/yolov5-master/datasets/images/test')
    project = Path('/home/zhengzf/.virtualenvs/yolov13/runs/pred')
    name = 'exp声呐-12'
    save_dir = _auto_increment_dir(project, name)

    # 选择设备
    device = '0' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = YOLO(weights)

    # 收集图片
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = [p for p in source_dir.rglob('*') if p.suffix.lower() in exts]
    if not images:
        print(f'未在 {source_dir} 找到图片文件')
        return

    # 绘制参数
    box_thickness = 3
    font_scale = 0.8
    font_thickness = 2
    text_padding = 4

    for img_path in images:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f'跳过无法读取的文件: {img_path}')
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.5,
            iou=0.5,
            device=device,
            verbose=False
        )
        res = results[0]
        names = res.names if hasattr(res, 'names') else model.names

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
                class_name = names.get(c, str(c)) if isinstance(names, dict) else names[c]
                color = _color_for_class(class_name)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(im0, (x1, y1), (x2, y2), color, box_thickness)

                label = f'{class_name} {s:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                tx = max(x1, text_padding)
                ty = max(y1 - th - text_padding, text_padding)
                cv2.rectangle(im0, (tx, ty), (tx + tw + text_padding * 2, ty + th + text_padding * 2), color, -1)
                cv2.putText(im0, label, (tx + text_padding, ty + th + text_padding),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        out_path = save_dir / f'{img_path.stem}_pred.jpg'
        cv2.imwrite(str(out_path), im0)
        print(f'Saved {out_path}')

    print(f'Done. All results saved to {save_dir}')


if __name__ == '__main__':
    run_folder_infer()