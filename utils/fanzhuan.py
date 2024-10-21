# import cv2
# import torch
#
# # 指定训练好的权重文件路径q
# weights_path = r'D:\lx\yolov5-master\runs\train\exp16\weights\best.pt'
#
# # 加载YOLOv5模型并指定权重文件
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
#
# # 设置模型为评估模式
# model.eval()
#
# # 打开摄像头
# cap = cv2.VideoCapture(0)  # 使用数字0表示默认摄像头，如果是其他设备则需要相应的编号
#
# while True:
#     # 读取一帧图像
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # 转换图像格式以便YOLOv5模型处理
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 进行物体检测
#     results = model(frame_rgb)
#
#     # 获取预测结果
#     predictions = results.pandas().xyxy[0]
#
#     # 计算所有预测框的像素值
#     total_pixels = 0
#     for _, row in predictions.iterrows():
#         width = row['xmax'] - row['xmin']
#         height = row['ymax'] - row['ymin']
#         bbox_area = int(width * height)
#         total_pixels += bbox_area
#
#         # 绘制预测框
#         cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
#                       (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
#
#         # 可以在这里添加类别标签
#         label = f"{row['name']} {row['confidence']:.2f}"
#         cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # 在这里可以打印总的像素值或者其他操作
#     print(f"Total pixels covered by bounding boxes: {total_pixels}")
#
#     # 显示带有预测框的图像
#     cv2.imshow('frame', frame)
#
#     # 按'q'键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch

def detect_objects_and_calculate_pixels(weights_path, source=0):

    # 加载YOLOv5模型并指定权重文件
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    # 设置模型为评估模式
    model.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(source)  # 使用数字0表示默认摄像头，如果是其他设备则需要相应的编号

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        if not ret:
            break

        # 转换图像格式以便YOLOv5模型处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行物体检测
        results = model(frame_rgb)

        # 获取预测结果
        predictions = results.pandas().xyxy[0]

        # 计算所有预测框的像素值
        total_pixels = 0
        for _, row in predictions.iterrows():
            width = row['xmax'] - row['xmin']
            height = row['ymax'] - row['ymin']
            bbox_area = int(width * height)
            total_pixels += bbox_area

            # 绘制预测框
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)

            # 可以在这里添加类别标签
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 在这里可以打印总的像素值或者其他操作
        print(f"Total pixels covered by bounding boxes: {total_pixels}")

        # 显示带有预测框的图像
        cv2.imshow('frame', frame)

        # 按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 示例调用
if __name__ == '__main__':
    weights_path = r'D:\lx\yolov5-master\runs\train\exp16\weights\best.pt'
    source = 0  # 默认摄像头
    detect_objects_and_calculate_pixels(weights_path, source)





