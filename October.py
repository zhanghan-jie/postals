class ObjectDetectionOutput:
    def __init__(self, img_shape):
        self.img_shape = img_shape  # 图片的形状（高度，宽度）

    def calculate_dimensions(self, detections):
        results = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]  # 假设det是[x1, y1, x2, y2, ...]的形式
            width = x2 - x1
            height = y2 - y1
            pixel_coordinates = (x1, y1, width, height)
            results.append(pixel_coordinates)
        return results
