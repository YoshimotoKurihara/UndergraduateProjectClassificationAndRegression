# coding:utf-8
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTrainer:
    def __init__(self, model, dataloaders, optimizer, device, task):
        self.model = model

        self.device = device
        self.model = self.model.to(device)
        self.task = task

        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss2d()

        self.epoch=0

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []        

    def run(self, epoch_num):
        self.master_bar = master_bar(range(epoch_num))
        for epoch in self.master_bar:
            self.epoch += 1
            self.train()
            self.test()
            self.save()
            self.draw_graph()
            if self.task == "Classification":
                self.draw_acc()
    
    def train(self):
        self.iter(train=True)
    
    def test(self):
        self.iter(train=False)

    def iter(self, train):
        if train:
            self.model.train()
            dataloader = self.dataloaders.train
        else:
            self.model.eval()
            dataloader = self.dataloaders.test
        
        total_loss = 0.
        total_acc = 0.

        data_iter = progress_bar(dataloader, parent=self.master_bar)
        for i, batch in enumerate(data_iter):
            image_list = batch["image"].to(self.device)
            if self.task == "Classification":
                class_list = batch["class"].to(self.device)
                # forward
                classes = self.model(image_list)

                # calc loss
                class_list= class_list.view(-1)
                loss = self.nll_loss(classes, class_list)
            if self.task == "Regression":
                pos_list = batch["pos"].to(self.device)
                positions = self.model(image_list)
                pos_list=pos_list.view(-1,6)
                loss = self.mse_loss(positions, pos_list)

            # backward
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad
            total_loss += loss.item()

            if self.task == "Classification":
                class_predicts=torch.argmax(classes,dim=1)
                acc = (class_predicts == class_list).sum().item() / len(class_list)
                total_acc += acc


        if train:
            self.train_loss_list.append(total_loss/ (i + 1))
            if self.task == "Classification":
                self.train_acc_list.append(total_acc/ (i + 1))
        else:
            self.val_loss_list.append(total_loss/ (i + 1))
            if self.task == "Classification":
                self.val_acc_list.append(total_acc/ (i + 1))

        train = "train" if train else "test" 
        print("[Info] epoch {}@{}: loss = {}".format(self.epoch, train, total_loss/ (i + 1)))
        if self.task == "Classification":
            print("[Info] epoch {}@{}: loss = {}, acc = {}".format(self.epoch, train, total_loss/ (i + 1), total_acc/(i + 1)))
    
    def save(self, out_dir="./output"):
        model_state_dict = self.model.state_dict()

        checkpoint = {
            "model": model_state_dict,
            "epoch": self.epoch,
        }

        if self.task == "Classification":
            model_name = "pose_acc_{acc:3.3f}.chkpt".format(
                acc = self.val_acc_list[-1]
            )
        if self.task == "Regression":
            model_name = "regression.chkpt"
        torch.save(checkpoint, model_name)
    
    def draw_graph(self):
        x = np.arange(self.epoch)
        y = np.array([self.train_loss_list, self.val_loss_list]).T
        plt.figure()
        plots = plt.plot(x, y)
        plt.legend(plots, ("train", "test"), loc="best", framealpha=0.25, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.tight_layer()
        graph_name = {
            "Classification" : "graph_loss.png",
            "Regression" : "graph_reg.png"
        }
        plt.savefig(graph_name[self.task])
        # plt.show()
    def draw_acc(self):
        x = np.arange(self.epoch)
        acc = np.array([self.train_acc_list, self.val_acc_list]).T
        plt.figure()
        plots_acc = plt.plot(x, acc)
        plt.legend(plots_acc, ("train", "test"), loc="best", framealpha=0.25, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.savefig("graph_acc.png")
