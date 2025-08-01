# -*- encoding: utf-8 -*-

# 关于pairloss: PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals

import torch
from torch import nn

class PairLoss(nn.Module):

    def __init__(self, max_iter=1000, eta=1e-5, upper_threshold=0.9, lower_threshold=0.5):
        super(PairLoss, self).__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.threshold = upper_threshold

    def forward(self, source_label, source_logits, target_logits):
        sim_matrix = self.get_cos_similarity_distance(source_logits)
        sim_matrix_target = self.get_cos_similarity_distance(target_logits)
        # print(sim_matrix.size())
        estimated_sim_truth = self.get_cos_similarity_distance(source_label)# one-hot编码，只有标签相同的才为1
        estimated_sim_truth_target = self.get_cos_similarity_by_threshold(sim_matrix_target) #目标域的标签，大于threshold的为1
        
        # pariwise loss for source domain
        bce_loss=-(torch.log(sim_matrix+self.eta)*estimated_sim_truth)-(1-estimated_sim_truth)*torch.log(1-sim_matrix+self.eta)
        cls_loss = torch.mean(bce_loss)

        bce_loss_target=-(torch.log(sim_matrix_target+self.eta)*estimated_sim_truth_target)-(1-estimated_sim_truth_target)*torch.log(1-sim_matrix_target+self.eta)
        # 去掉非法的target loss
        indicator, nb_selected = self.compute_indicator(sim_matrix_target)
        cluster_loss=torch.sum(indicator*bce_loss_target)/nb_selected

        return cls_loss, cluster_loss
        
    def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True) #计算每行的L2范数
        # (batch_size, num_clusters)
        features = features / features_norm #每行的每个都除以L2范数
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1)) #计算出两两之间的余弦相似度
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar, 
                                 dissimilar)#这里用threshold还是upper_threshold呢，结果是一模一样的
        return sim_matrix
    
    def compute_indicator(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        selected = torch.tensor(1, dtype=dtype, device=device)
        not_selected = torch.tensor(0, dtype=dtype, device=device)
        w2 = torch.where(cos_dist_matrix < self.lower_threshold, selected, not_selected)
        w1 = torch.where(cos_dist_matrix > self.upper_threshold, selected, not_selected)
        w = w1 + w2
        nb_selected = torch.sum(w)
        return w, nb_selected
    
    def update_threshold(self, epoch):
        # 更新threshold
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
        #        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold - eta
            self.lower_threshold = self.lower_threshold + eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold = (self.upper_threshold + self.lower_threshold) / 2
