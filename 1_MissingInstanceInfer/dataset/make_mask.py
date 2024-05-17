import numpy as np
from PIL import Image, ImageDraw
import math
import random


def IrregularBox(
    mask_size, #(w,h)
    bbox,
    min_num_vertex = 15,
    max_num_vertex = 30):
    
    W, H = mask_size
    mask = Image.new('L', (W, H), 0)
    x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cx, cy = int((x1+x2)/2) , int((y1+y2)/2)
    box_width = int(x2-x1)
    box_height = int(y2-y1)
    num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
    
    vertex = []
    for i in range(num_vertex):
        vertex.append((int(cx+np.random.randint(-(box_width//2), (box_width//2))), int(cy+np.random.randint(-(box_height//2), (box_height//2)))))

    draw = ImageDraw.Draw(mask)
    width = int(box_width * 0.5)
    # draw.rounded_rectangle(xy=[x1,y1,x2,y2], radius=10, fill=1)
    draw.rectangle(xy=[x1,y1,x2,y2], fill=1)
    draw.line(vertex, fill=1, width=width)
    
    for v in vertex:
        draw.ellipse((v[0] - width//2,
                      v[1] - width//2,
                      v[0] + width//2,
                      v[1] + width//2),
                     fill=1)
        
    mask = np.asarray(mask, np.uint8)
    
    return mask

def MultipleIrregularBox(
    mask_size,
    bboxes,
    min_num_vertex = 15,
    max_num_vertex = 30):

    W, H = mask_size
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cx, cy = int((x1+x2)/2) , int((y1+y2)/2)
        box_width = int(x2-x1)
        box_height = int(y2-y1)
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        
        vertex = []
        for i in range(num_vertex):
            vertex.append((int(cx+np.random.randint(-(box_width//2), (box_width//2))), int(cy+np.random.randint(-(box_height//2), (box_height//2)))))

        width = int(box_width * 0.5)
        # draw.rounded_rectangle(xy=[x1,y1,x2,y2], radius=10, fill=1)
        draw.rectangle(xy=[x1,y1,x2,y2], fill=1)
        draw.line(vertex, fill=1, width=width)
        
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                        v[1] - width//2,
                        v[0] + width//2,
                        v[1] + width//2),
                        fill=1)
        
    mask = np.asarray(mask, np.uint8)
    
    return mask

def MakeMaskSingleBox(mask_size, bbox):
    mask = 1 - IrregularBox(mask_size, bbox) ## zero for the missing region, one for the remaining region
    return mask[np.newaxis, ...].astype(np.float32) 

def MakeMaskMultipleBox(mask_size, bboxes):
    mask = 1 - MultipleIrregularBox(mask_size, bboxes) ## zero for the missing region, one for the remaining region
    return mask[np.newaxis, ...].astype(np.float32) 