# -*- encoding: utf-8 -*-

# 跑一下基本的模型，做一个基线

import os
import numpy as np
import torch
from torch import nn
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from training.config import get_parser
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset
from models import Base
from trainers import BaseTrainer

def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()


class CustomDataset(TensorDataset):
    def __init__(self, d1, d2):
        super(CustomDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx]
    
    def data(self):
        return self.d1
    def label(self):
        return self.d2

def load_seed(args, target, source):

    # EEG, Label, Group = SEEDFeatureDataset(args.path, session=args.session).data()
    # Label += 1
    EEG, Label, Group = SEEDIVFeatureDataset(args.path, session=args.session).data() #SEEDIV的Label不用加1
    EEG = EEG.reshape(-1, 310)
    tGroup = Group[:, 2] - 1 # 影片的group
    sGroup = Group[:, 1]
    

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(sGroup):
        EEG[sGroup==i] = min_max_scaler.fit_transform(EEG[sGroup == i])

    one_hot_mat = np.eye(len(Label), args.num_of_class)[Label].astype("float32") #seediv有4类

    # 一对一，每个受试者都是一个源域，每个目标域对应14个源域，共进行14*15次训练
    # 获得目标域数据
    target_features = torch.from_numpy(EEG[sGroup==target]).type(torch.Tensor)
    target_labels = torch.from_numpy(one_hot_mat[sGroup==target])
    torch_dataset_target = CustomDataset(target_features, target_labels)
    
    # 获得源域数据
    source_features = torch.from_numpy(EEG[sGroup==source]).type(torch.Tensor)
    source_labels = torch.from_numpy(one_hot_mat[sGroup==source])
    torch_dataset_source = CustomDataset(source_features, source_labels)

    return torch_dataset_source, torch_dataset_target

def get_model_utils(args):
    # 模型
    base_params = {
        "num_of_class" : args.num_of_class,
    }
    params = {
        "transfer_loss_type" : args.transfer_loss_type,
        "max_iter" : args.max_iter
    }

    combined_params = {**base_params, **params}
    model = Base(**combined_params).cuda()

    # 优化器
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None
    # 训练器
    trainer_params = {
        "lr_scheduler" : scheduler,
        "batch_size" : args.batch_size,
        "n_epochs" : args.n_epochs,
        "transfer_loss_weight" : args.transfer_loss_weight,
        "early_stop" : args.early_stop,
        "tmp_saved_path" : args.tmp_saved_path,
        "log_interval" : args.log_interval,
    }
    trainer = None
    trainer = BaseTrainer(
        model, 
        optimizer, 
        **trainer_params
    )
    return trainer


def train(target, source, args):

    # 每一个受试者都重新定义seed
    setup_seed(args.seed)
    
    # 记得换
    cur_target_saved_path = os.path.join(args.tmp_saved_path, f'logs_{args.model_type}_{args.transfer_loss_type}_iv',f'session_{args.session}', f'source{source}_to_target{target}')
    create_dir_if_not_exists(cur_target_saved_path)

    # 获取数据
    dataset_source, dataset_target = load_seed(args, target=target, \
                                               source=source)
    loader_source = DataLoader(
            dataset=dataset_source,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )
    loader_target = DataLoader(
            dataset=dataset_target,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )

    # 定义max_iter
    setattr(args, "max_iter", 1000) # 按理说应该等于上式，但是原始代码定义为了n_epochs

    # 获得训练器
    trainer = get_model_utils(args)
    
    # 训练
    best_acc, np_log = trainer.train(loader_source, loader_target)    
    
    # 保存模型
    if args.saved_model:
        torch.save(trainer.get_model_state(), os.path.join(cur_target_saved_path, f"last.pth")) #当前目标域的最终的模型参数
        torch.save(trainer.get_best_model_state(), os.path.join(cur_target_saved_path, f"best.pth")) #当前目标域最好的模型参数,应该是属于一个epoch的
    np.savetxt(os.path.join(cur_target_saved_path, f"logs.csv"), np_log, delimiter=", ",  fmt='%.4f')
    return best_acc



def main(args):
    setup_seed(args.seed)
    # 用来测试不同的迁移损失函数

    
    for target in range(1, 16):# 15个目标域
        best_acc_mat = []
        print(f"target: {target} start training...")
        for source in range(1, 16):# 每个目标域14个源域分别训练
            if target == source:
                continue
            best_acc = train(target, source, args)
            best_acc_mat.append(best_acc)
            print(f"target: {target}, source: {source}, best_acc: {best_acc}")
        
        

def create_dir_if_not_exists(path):
    # 构建完整的路径
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    main(args)
    # test(args)
