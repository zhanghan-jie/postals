import torch
import torch.nn as nn
from models.yolo import Detect, Model
from models.experimental import attempt_load
from models.experimental import Ensemble  # 确保 Ensemble 类被正确导入


class CustomYOLOv5(nn.Module):
    def __init__(self, model_path=None, device=None, inplace=True, fuse=True):
        super(CustomYOLOv5, self).__init__()
        self.device = device or torch.device('cpu')
        self.model = None

        if model_path is not None:
            # 尝试加载模型权重
            self.model = self.attempt_load(model_path, device=self.device, inplace=inplace, fuse=fuse)
            if self.model is None:
                raise RuntimeError("Failed to load the model.")

        if self.model is not None:
            self.model.eval()  # 设置模型为评估模式

    def attempt_load(self, weights, device=None, inplace=True, fuse=True):
        """
        Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.
        """
        model = attempt_load(weights, device=device, inplace=inplace, fuse=fuse)
        return model

    def forward(self, img):
        # 进行物体检测
        with torch.no_grad():
            results = self.model(img)  # 假设返回的是预测结果

            # 打印结果的类型和内容以调试
            print(f"Results type: {type(results)}")
            print(f"Results content: {results}")

        # 获取预测框
        if isinstance(results, tuple):
            detections = results[0].cpu().numpy()  # 获取第一个图像的检测结果
        else:
            detections = results.cpu().numpy()  # 直接获取检测结果

        # 计算每个检测框的像素值
        pixel_values = []
        for det in detections:
            # 提取边界框的坐标
            x1, y1, x2, y2 = det[:4]

            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1

            # 确保宽度和高度是标量值
            if isinstance(width, torch.Tensor):
                width = width.item()
            if isinstance(height, torch.Tensor):
                height = height.item()

            # 计算边界框面积
            bbox_area = int(width * height)
            pixel_values.append(bbox_area)

        return results, pixel_values