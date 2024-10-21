# # -*- coding: utf-8 -*-
# import xml.etree.ElementTree as ET
# import os
# from os import getcwd
#
# sets = ['train', 'val', 'test']
# classes = ["car", "people"]  # 改成自己的类别
# abs_path = os.getcwd()
# print(abs_path)
#
#
# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = (box[0] + box[1]) / 2.0 - 1
#     y = (box[2] + box[3]) / 2.0 - 1
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return x, y, w, h
#
#
# def convert_annotation(image_id):
#     in_file = open('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCDATA/Annotations/%s.xml' % (image_id), encoding='UTF-8')
#     out_file = open('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCDATA/labels/%s.txt' % (image_id), 'w')
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#     for obj in root.iter('object'):
#         difficult = obj.find('difficult').text
#         # difficult = obj.find('Difficult').text
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult) == 1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#              float(xmlbox.find('ymax').text))
#         b1, b2, b3, b4 = b
#         # 标注越界修正
#         if b2 > w:
#             b2 = w
#         if b4 > h:
#             b4 = h
#         b = (b1, b2, b3, b4)
#         bb = convert((w, h), b)
#         out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
#
#
# wd = getcwd()
# for image_set in sets:
#     if not os.path.exists('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/labels/'):
#         os.makedirs('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/labels/')
#     image_ids = open('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
#
#     if not os.path.exists('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/dataSet_path/'):
#         os.makedirs('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/dataSet_path/')
#
#     list_file = open('dataSet_path/%s.txt' % (image_set), 'w')
#     # 这行路径不需更改，这是相对路径
#     for image_id in image_ids:
#         list_file.write('C:/Users/寒杰/OneDrive/桌面/yolov5-master/VOCData/images/%s.jpg\n' % (image_id))
#         convert_annotation(image_id)
#     list_file.close()
from moviepy.editor import VideoFileClip

def convert_flv_to_mp4(input_flv_path, output_mp4_path):
    # 加载 FLV 文件
    clip = VideoFileClip(input_flv_path)
    # 将 FLV 文件保存为 MP4 文件
    clip.write_videofile(output_mp4_path, codec='libx264')

# 指定输入和输出文件路径
input_flv_path =r'D:\BaiduNetdiskDownload\行人检测测试视频.flv'
output_mp4_path = r'D:\BaiduNetdiskDownload\output.mp4'

# 转换文件
convert_flv_to_mp4(input_flv_path, output_mp4_path)