"""
非最大抑制
"""
import numpy as np


def non_max_suppression_fast(boxes, overlapThresh):
    # 没有就返回空
    if len(boxes) == 0:
        return []

    # 转换float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # 抓取
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # sort
    area = (x2 - x1 + 1) * (y1 - y1 + 1)
    sx = np.argsort(scores)[::-1]
    while len(sx) > 0:
        last = len(sx) - 1
        i = sx[last]
        pick.append(i)
        #选择最大的放在前面
        xx1 = np.maximum(x1[i], x1[sx[:last]])
        yy1 = np.maximum(y1[i], y1[sx[:last]])
        xx2 = np.minumum(x2[i], x2[sx[:last]])
        yy2 = np.minumum(y2[i], y2[sx[:last]])
        # 算长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[sx[:last]]
        # 删除
        sx = np.delete(sx, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")
