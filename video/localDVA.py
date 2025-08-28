import cv2
import numpy as np

import torch
import torch.nn as nn

import copy
import time

# video functions

def frame_generator(cap, W, H):
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (W, H))
            yield frame
        else:
            break

def get_diff_mask(frame, prev_frame, threshold=0.1):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.absdiff(grey, prev_grey)
    _, thresh = cv2.threshold(mask, 255 * threshold, 255, cv2.THRESH_BINARY)
    return thresh

def get_bounding_box(mask, threshold=0.1):
    _, thresh = cv2.threshold(mask, 255 * threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # pick the largest contour
    # cnt = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(cnt)
    # get the bounding rect of all contours
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return x, y, w, h

# load model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.AdaptiveAvgPool2d((32, 32)),
    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 10)
)

def make_delta_model(model):
    dmodel = copy.deepcopy(model)
    for i, lyr in enumerate(dmodel):
        if isinstance(lyr, nn.Conv2d):
            lyr.bias = None
            pad = lyr.padding
            if isinstance(pad, int):
                pad = (pad, pad)
            lyr.padding = (max(pad[0], lyr.kernel_size[0] - 1), max(pad[1], lyr.kernel_size[1] - 1))
        elif isinstance(lyr, nn.Linear):
            lyr.bias = None

dmodel = make_delta_model(model)

# basic utils

def frame2tensor(frame):
    frame = frame.transpose((2, 0, 1))
    frame = torch.from_numpy(frame)
    frame = frame.float()
    frame = frame / 255
    return frame

def get_tensor_in_box(data:torch.Tensor, x, y, w, h):
    return data[..., y:y+h, x:x+w]

# normal processing

@torch.no_grad()
def full_process(model:torch.nn, frame:torch.Tensor):
    # d = frame2tensor(frame).unsqueeze(0)
    d = frame
    buffer = [d]
    with torch.no_grad():
        for i, lyr in enumerate(model):
            d = lyr(d)
            buffer.append(d)
    return buffer

# delta processing (utils)

def map_conv_incoord_to_outcoord(conv, x, y):
    x, y = x + conv.padding[0], y + conv.padding[1]
    x, y = x // conv.stride[0], y // conv.stride[1]
    return x, y

def map_conv_delta_incoord_to_outcoord(conv, x, y):
    x, y = x + conv.padding[0], y + conv.padding[1]
    x, y = x // conv.stride[0], y // conv.stride[1]
    x, y = x - conv.kernel_size[0] + 1, y - conv.kernel_size[1] + 1
    return x, y

def map_adaptive_pool_incoord_to_outcoord(pool, w, h, x, y):
    x, y = x // (w / pool.output_size[0]), y // (h / pool.output_size[1])
    x, y = int(x), int(y)
    return x, y

def map_pool_incoord_to_outcoord(pool, x, y):
    x, y = x + pool.padding[0] // pool.stride[0], y + pool.padding[1] // pool.stride[1]
    return x, y


def delta_conv(lyr_o, lyr_d, prev, d, b):
    # lyr_o: original layer
    # lyr_d: delta layer
    # prev: previous output using the original layer
    # d: delta data
    # b: bounding box
    d = lyr_d(d)
    ox, oy = map_conv_delta_incoord_to_outcoord(lyr_o, b[0], b[1])
    oh, ow = d.shape[2], d.shape[3]
    # print(f'  {b} -> {(ox, oy, ow, oh)}, {d.shape}')
    # cut off those out of box
    original_height, original_width = prev.shape[2:]
    cut_x_0 = -ox if ox < 0 else 0
    cut_y_0 = -oy if oy < 0 else 0
    cut_x_1 = ox + ow - original_width if ox + ow > original_width else 0
    cut_y_1 = oy + oh - original_height if oy + oh > original_height else 0
    if cut_x_0 != 0 or cut_y_0 != 0 or cut_x_1 != 0 or cut_y_1 != 0:
        d = d[..., cut_y_0:oh-cut_y_0-cut_y_1, cut_x_0:ow-cut_x_0-cut_x_1]
        # print(f'  cut to {d.shape}')
    b = (ox, oy, ow, oh)
    return d, b

def delta_relu(lyr_o, lyr_d, prev, d, b):
    return lyr_d(d), b

def delta_maxpool(lyr_o, lyr_d, prev, d, b):
    # padding: first int is used for the height dimension, and the second int for the width dimension
    x0_q, x0_r = divmod(b[0] + lyr_d.padding[0], lyr_d.stride[0])
    y0_q, y0_r = divmod(b[1] + lyr_d.padding[1], lyr_d.stride[1])
    # x1_q, x1_r = divmod(b[0] + b[2] + lyr_d.padding[0], lyr_d.stride[0])
    # y1_q, y1_r = divmod(b[1] + b[3] + lyr_d.padding[1], lyr_d.stride[1])
    if x0_r != 0 or y0_r != 0:
        lyr_d.padding = (lyr_d.padding[0], lyr_d.padding[1])
    lyr_d.ceil_mode = True
    d = lyr_d(d)
    # ox, oy = map_pool_incoord_to_outcoord(model[i], b[0], b[1])
    ox, oy = x0_q, y0_q
    oh, ow = d.shape[2], d.shape[3]
    # print(f'  {b} -> {(ox, oy, ow, oh)}, {d.shape}')
    return d, (ox, oy, ow, oh)

def delta_adaptive_pool(lyr_o, lyr_d, prev, d, b):
    h, w = prev.shape[2], prev.shape[3]
    ox, oy = map_adaptive_pool_incoord_to_outcoord(lyr_o, w, h, b[0], b[1])
    ox2, oy2 = map_adaptive_pool_incoord_to_outcoord(lyr_o, w, h, b[0]+b[2], b[1]+b[3])
    ow, oh = ox2-ox, oy2-oy
    lyr = nn.AdaptiveAvgPool2d((oh, ow))
    d = lyr(d)
    # print(f'  {b} -> {(ox, oy, ow, oh)}, {d.shape}')
    b = (ox, oy, ow, oh)
    return d, b

def delta_flatten(lyr_o, lyr_d, prev, d, b):
    ox, oy, ow, oh = b
    # print(f'  {prev.shape}, {b}, {d.shape}, {prev[..., oy:oy+oh, ox:ox+ow].shape}')
    # tmp = prev_buffer[i].clone()
    # tmp[..., oy:oy+oh, ox:ox+ow] += d
    tmp = torch.zeros_like(prev)
    tmp[..., oy:oy+oh, ox:ox+ow] = d
    d = lyr_d(tmp)
    b = None
    return d, b

def delta_linear(lyr_o, lyr_d, prev, d, b):
    d = lyr_d(d)
    b = None
    return d, b

# delta processing

# def bgr2gray(t: torch.Tensor):
#     return t[..., 0] * 0.299 + t[..., 1] * 0.587 + t[..., 2] * 0.114

@torch.no_grad()
def delta_process(model:torch.nn, dmodel:torch.nn, prev_buff:list, frame:torch.nn, bb:tuple):
    # buff = [d]
    delta = get_tensor_in_box(d-prev_buff[0], *bb).unsqueeze(0)
    for i, lyr in enumerate(dmodel):
        if isinstance(lyr, nn.Conv2d):
            d, bb = delta_conv(lyr, prev_buff[i], delta, bb)
        elif isinstance(lyr, nn.ReLU):
            d, bb = delta_relu(lyr, prev_buff[i], delta, bb)
        elif isinstance(lyr, nn.MaxPool2d):
            d, bb = delta_maxpool(lyr, prev_buff[i], delta, bb)
        elif isinstance(lyr, nn.AdaptiveAvgPool2d):
            d, bb = delta_adaptive_pool(lyr, prev_buff[i], delta, bb)
        elif isinstance(lyr, nn.Flatten):
            d, bb = delta_flatten(lyr, prev_buff[i], delta, bb)
        elif isinstance(lyr, nn.Linear):
            d, bb = delta_linear(lyr, prev_buff[i], delta, bb)
        else:
            print(f'  {i+1}-th {lyr} is not supported')
    return d


# main

def run_video(cap:cv2.VideoCapture, model:torch.nn, W:int, H:int, num_frame:int=None,
              diff_threshold:float=0.05, omit_threshold=0.01, renew_threshold:int=0.6):
    omit_threshold = int(omit_threshold * W * H)
    renew_threshold = int(renew_threshold * W * H)
    
    dmodel = make_delta_model(model)
    t = time.time()
    frame = next(frame_generator(cap, W, H))
    d = frame2tensor(frame).unsqueeze(0)
    buffer = full_process(model, d)
    t = time.time() - t
    
    results = [buffer[-1]]
    times = [t]
    bbs = [None]
    prev_frame = frame
    while num_frame is None or num_frame > 0:
        t = time.time()
        frame = next(frame_generator(cap, W, H))
        mask = get_diff_mask(frame, prev_frame, 0.05)
        # print((mask!=0).sum())
        bb = get_bounding_box(mask)
        # print(bb)
        if bb is None or bb[2]*bb[3] < omit_threshold:
            res = buffer[-1]
        elif bb[2]*bb[3] > renew_threshold:
            buffer = full_process(model, d)
        else:
            d = frame2tensor(frame).unsqueeze(0)
            res = delta_process(model, dmodel, buffer, d, bb)
        times.append(time.time() - t)
        bbs.append(bb)
        results.append(buffer[-1])
    return results, times, bbs

def main():
    video_path='E:/Data/video/pose_video_dataset/002_dance.mp4'
    cap = cv2.VideoCapture(video_path)
    # W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W, H = 640, 360
    
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((32, 32)),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 10)
    )
    
    run_video(cap, model, W, H, 0.05)
    
    