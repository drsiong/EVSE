# -*- encoding: utf-8 -*-
# PRPL的模型实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from loss_funcs import PairLoss, TransferLoss

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):#比论文中少了一层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class LabelClassifier(nn.Module):
    def __init__(self, 
                 num_of_class: int = 3, 
                 low_rank: int = 32, 
                 max_iter: int = 1000, 
                 upper_threshold: float = 0.9, 
                 lower_threshold: float = 0.5):
        
        super(LabelClassifier, self).__init__()

        # 定义可训练参数
        self.U = nn.Parameter(torch.randn(low_rank, 64), requires_grad=True)
        self.V = nn.Parameter(torch.randn(low_rank, 64), requires_grad=True)
        self.P = torch.randn(num_of_class, 64) #3*64
        self.stored_mat = torch.matmul(self.V, self.P.T) # 32*64 64*3

        # 定义参数
        self.max_iter = max_iter
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.threshold = upper_threshold
        self.num_of_class = num_of_class

        self.register_buffer('cluster_label', torch.zeros(num_of_class))
        

    def forward(self, feature): #进来样本特征，得到交互特征，即对每个样本的预测，做的是一个聚簇的操作
        preds = torch.matmul(torch.matmul(self.U, feature.T).T, self.stored_mat)   # 32*64 64*batch_size   32*3 = batch_size*3
        logits = F.softmax(preds, dim=1)#这里只是得到了每个样本属于某个聚簇？
        return logits 

    def update_P(self, source_feature, source_label): # 3*batch_size batch_size*64
        self.P = torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0)) + torch.eye(self.num_of_class).cuda()),
                              torch.matmul(source_label.T, source_feature)) #后者是每类加权和，前者是每类的个数，前者的逆与后者作矩阵乘法，得到每类的平均值，即miu_c
        self.stored_mat = torch.matmul(self.V, self.P.T) 

    
    def update_cluster_label(self, source_feature, source_label):#只针对源域更新
        self.eval()
        with torch.no_grad():
            logits = self.forward(source_feature)
            source_cluster = np.argmax(logits.cpu().detach().numpy(), axis=1) #找到每个样本的聚簇，一维数组
            source_label = np.argmax(source_label.cpu().numpy(), axis=1) #每个样本的标签，一维数组
            for i in range(self.num_of_class):
                # 获取当前聚簇的样本
                samples_in_cluster_index = np.where(source_cluster == i)[0] #找到样本中cluster等于i的样本的索引，【0】是where的特性
                label_for_samples = source_label[samples_in_cluster_index] #根据索引得到样本的真实标签

                # 对当前聚簇的样本进行分析，获得该聚簇类别数最多的类别，并定义
                if len(label_for_samples) == 0:#没有样本的聚类是第0类
                    self.cluster_label[i] = 0
                else:#有样本的聚类
                    label_for_current_cluster = np.argmax(np.bincount(label_for_samples))#bincount统计真实标签的出现次数，然后选取最大的那个出现次数
                    self.cluster_label[i] = label_for_current_cluster #标量，也就是获得了聚簇是i的最多的样本类别/标签

    def predict(self, feature):#对源域和目标域是一样的预测吗，由于存在域对抗，所以源域和目标域的分布逐渐相同，因此使用了源域的cluster_label
        with torch.no_grad():
        
            logits = F.softmax(self.forward(feature), dim=1)#这个做的是一个聚簇的操作，并不知道这个聚簇的标签
            cluster = np.argmax(logits.cpu().numpy(), axis=1)
            preds = np.array([self.cluster_label[i] for i in cluster])#这里才是聚簇属于的标签，是真正的预测
        return preds
    
    def get_parameters(self):
        params = [
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
        ]
        return params



class PRPL(nn.Module):
    def __init__(self, 
                 num_of_class: int = 3, 
                 max_iter: int = 1000,
                 low_rank: int = 32, 
                 transfer_loss_type: str="dann",
                 upper_threshold: float = 0.9, 
                 lower_threshold: float = 0.5,
                 **kwargs):
        super(PRPL, self).__init__()
        self.max_iter = max_iter
        self.fea_extrator = FeatureExtractor(310, 64, 64)

        # 去掉下面那个代码会导致降低1.2%的准确率，属于噪声叭
        # self.fea_extrator_g = FeatureExtractor(310, 64, 64) #这个会有影响
        self.classifier = LabelClassifier(num_of_class = num_of_class, 
                                               low_rank = low_rank, 
                                               max_iter = max_iter, 
                                               upper_threshold = upper_threshold, 
                                               lower_threshold = lower_threshold)
        
        self.transfer_loss_type = transfer_loss_type
        self.num_of_class = num_of_class
        self.pair_loss = PairLoss(max_iter=max_iter)
        transfer_loss_args = {
            "loss_type" : self.transfer_loss_type,
            "max_iter" : self.max_iter,
            "num_class" : self.num_of_class,
            **kwargs
        }

        self.transfer_loss = TransferLoss(**transfer_loss_args) #对抗域损失/迁移损失

    def forward(self, source, target, source_label):
        batch_size = source.size(0)
        source_feature = self.fea_extrator(source)
        target_feature = self.fea_extrator(target)

        self.classifier.update_P(self.fea_extrator(source), source_label)
        # 直接使用source_feature和额外的进行一次特征的提取准确率会不一致，且准确率会下降
        # self.classifer.update_P(source_feature, source_label)
        
        source_logits = self.classifier(source_feature)
        target_logits = self.classifier(target_feature)

        clf_loss, cluster_loss = self.pair_loss(source_label, source_logits, target_logits)
        
        P_loss=torch.norm(torch.matmul(self.classifier.P.T,self.classifier.P)-torch.eye(64).to(source.device),'fro')

        kwargs = {}
        if self.transfer_loss_type == "lmmd":
            kwargs["source_label"] = source_label
            target_clf = self.classifier(target_feature)
            kwargs["target_logits"] = F.softmax(target_clf, dim=1)
        elif self.transfer_loss_type == "daan":
            source_clf = self.classifier(source_feature)
            kwargs['source_logits'] = F.softmax(source_clf, dim=1)
            target_clf = self.classifier(target_feature)
            kwargs['target_logits'] = F.softmax(target_clf, dim=1)
        
        trans_loss = self.transfer_loss(source_feature+0.005*torch.randn((batch_size,64)).to(source_feature.device), \
                                         target_feature+0.005*torch.randn((batch_size,64)).to(target_feature.device), **kwargs)

        return clf_loss, cluster_loss,  P_loss, trans_loss #source的loss，target的loss，source上避免过拟合加的loss，域对抗loss
    

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.fea_extrator(x)
            preds = self.classifier.predict(feature)
        return preds

    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.fea_extrator(x)
            output = self.classifier(feature)
            logits = F.softmax(output, dim=1)
        return logits

    def get_parameters(self) :
        params = [
            *self.fea_extrator.get_parameters(),
            *self.classifier.get_parameters(),
        ]
        if self.transfer_loss_type == "dann":
            params.append(
                {"params": self.transfer_loss.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
        elif self.transfer_loss_type == "daan":
            params.append(
                {'params': self.transfer_loss.loss_func.domain_classifier.parameters(), "lr_mult":1}
            )
            params.append(
                {'params': self.transfer_loss.loss_func.local_classifiers.parameters(), "lr_mult":1}
            )
        return params
    
    def epoch_based_processing(self, epoch, source_features, source_labels):
        
        self.pair_loss.update_threshold(epoch)
        self.classifier.update_cluster_label(self.fea_extrator(source_features), source_labels)
