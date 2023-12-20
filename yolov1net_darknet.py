# 코드 출처: https://github.com/sendeniz/yolov1-real-time-obj-detection/blob/main/models/yolov1net_darknet.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:48:38 2022

@author: sen
"""
import torch
import torch.nn as nn

# 컨볼루션 레이어의 출력 크기를 계산하기 위한 함수
def conv_params(input_h, n_padding, kernel_size, n_stride):
    return (input_h - kernel_size + 2*n_padding) / (n_stride) + 1

# 커널 크기를 입력으로 받아 패딩의 수를 계산하는 함수
def num_paddings(kernel_size):
    return (kernel_size -1) //2

# YOLOv1 모델이 정의된 클래스
class YoloV1Net(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1Net, self).__init__()

        # darknet을 백본으로 하는 네트워크
        # darknet backbone
        self.darknet = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels = 3, out_channels = 64, 
                      kernel_size = (7, 7) , stride = 2, 
                      padding = 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 2
            nn.Conv2d(in_channels = 64, out_channels = 192, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 3
            nn.Conv2d(in_channels = 192, out_channels = 128, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 128, out_channels = 256, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 4
            nn.Conv2d(in_channels = 512, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1, 
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 512, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 5 (first 4 conv layers)
            nn.Conv2d(in_channels = 1024, out_channels = 512, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 512, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
                                )
        # YOLO의 출력을 생성하는 부분으로, 마지막 부분에서 객체의 경계 상자와 클래스에 대한 예측 수행
        # yolo head
        self.yolov1head = nn.Sequential (
            # Block 5 (last two conv layers)
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 2,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
           
            # Block 6
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # prediction block
            nn.Flatten(),
            nn.Linear(in_features = 1024 * S * S, out_features = 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = 4096, out_features = S * S * (C + B * 5)),
            # reshape in loss to be (S,S, 30) with C + B * 5 = 30
            )
        
        self.random_weight_init()

    # 모델의 정방향 전파 수행
    def forward(self, x):
        x = self.darknet(x)
        x = self.yolov1head(x)
        return x

    # Conv2d 레이어의 가중치를 무작위로 초기화하는 메서드
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[0].weight.data.normal_(0, 0.01)
        
# 모델의 테스트 함수
def test ():
    model = YoloV1Net()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

testx, xdims = test()

