# coding:utf-8
#export PYTHONPATH=/usr/local/lib/python3.6/dist-packages
#毎回これを実行
#インタープリタはpython3.7.4のbase condaのやつを使う

#データ正規化　バッチ正規化　学習率のスケジューラ
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os

#同ディレクトリ内の別のファイルからのクラスの読み込み
from submodel import AlexNet, PositionNet
from subdataloader import Dataloaders
from subtrainer import MyTrainer

def main():
    IMAGE_PATH = "/home/gonken2019/Desktop/subProject/dataset45"#"/home/gonken2019/Desktop/subProject/images"#
    LABELS_PATH = "/home/gonken2019/Desktop/subProject/poseData45/"#"/home/gonken2019/Desktop/subProject/labels/"#
    BATCH_SIZE = 256 #こことsubmodel.py 85行目と113行目の最初の引数を変える 
    NUM_EPOCH = 20 #多くて20~30

    if torch.cuda.is_available():
        device = "cuda";
        print("[Info] Use CUDA")
    else:
        device = "cpu"
    model1 = AlexNet()
    model2 = PositionNet()
    dataloaders = Dataloaders(IMAGE_PATH, LABELS_PATH, BATCH_SIZE)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=0.00001, weight_decay=5e-4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.0001, weight_decay=5e-4)
    #lossがnanになるのはよくあるので、こういうときはoptimizerを変えるか学習率変えるかするといい

    trainer1 = MyTrainer(model1, dataloaders, optimizer1, device, "Classification")
    trainer2 = MyTrainer(model2, dataloaders, optimizer2, device, "Regression")

    trainer1.run(NUM_EPOCH)#
    trainer2.run(NUM_EPOCH)

main()#
