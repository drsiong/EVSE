# -*- encoding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from training.config import get_parser
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset
from models import Base, PRPL
from trainers import BaseTrainer, PRPLTrainer



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
        #        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
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
    Group = Group[:, 1]

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(Group):
        EEG[Group==i] = min_max_scaler.fit_transform(EEG[Group == i])

    one_hot_mat = np.eye(len(Label), args.num_of_class)[Label].astype("float32")
    
    # 获得目标域数据
    target_features = torch.from_numpy(EEG[Group==target]).type(torch.Tensor)
    target_labels = torch.from_numpy(one_hot_mat[Group==target])
    torch_dataset_target = CustomDataset(target_features, target_labels)
    
    source_features = torch.from_numpy(EEG[Group==source]).type(torch.Tensor)
    source_labels = torch.from_numpy(one_hot_mat[Group==source])
    torch_dataset_source = CustomDataset(source_features, source_labels)

    return torch_dataset_source, torch_dataset_target

    

def get_model_utils(args):

    base_params = {
        "num_of_class" : args.num_of_class,
    }
    params = {
        "transfer_loss_type" : args.transfer_loss_type,
        "max_iter" : args.max_iter,
        "lower_rank" : args.lower_rank,
        "upper_threshold" : args.upper_threshold,
        "lower_threshold" : args.lower_threshold
    }
    
    combined_params = {**base_params, **params}
    model = PRPL(
        **combined_params
    ).cuda()

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
    trainer = PRPLTrainer(
        model, 
        optimizer, 
        **trainer_params
    )
    return trainer


def train(target, source, args):
    
    # 每一个受试者都重新定义seed
    setup_seed(args.seed)

    cur_target_saved_path = os.path.join(args.tmp_saved_path, f'logs_{args.model_type}_{args.transfer_loss_type}_ivv',f'session_{args.session}' ,f'source{source}_to_target{target}')
    create_dir_if_not_exists(cur_target_saved_path)
    # 获取数据
    dataset_source, dataset_target = load_seed(args, target=target, \
                                               source=source )
    loader_source = DataLoader(
            dataset=dataset_source,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
    loader_target = DataLoader(
            dataset=dataset_target,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
    
    # 定义max_iter
    setattr(args, "max_iter", 1000) # 按理说应该等于上式，但是原始代码定义为了n_epochs

    # 获得训练器
    trainer = get_model_utils(args)

    # 训练模型
    best_acc, np_log = trainer.train(loader_source, loader_target)   

    # 保存模型
    if args.saved_model:
        torch.save(trainer.get_model_state(), os.path.join(cur_target_saved_path, f"last.pth"))
        torch.save(trainer.get_best_model_state(), os.path.join(cur_target_saved_path, f"best.pth"))
    np.savetxt(os.path.join(cur_target_saved_path, f"logs.csv"), np_log, delimiter=", ",  fmt='%.4f')
    return best_acc

def main(args):
    setup_seed(args.seed)
    # 用来测试不同的迁移损失函数

    for source in range(4, 16):
        best_acc_list = []
        print(f"source: {source} start training...")
        for target in range(1, 16):
            if source == target:
                continue
            best_acc = train(target, source, args)
            best_acc_list.append(best_acc)
            print(f"source: {source}, target: {target}, best_acc: {best_acc:.2f}")
        best_acc_list = np.array(best_acc_list)
        mean_acc = np.mean(best_acc_list)
        std_acc = np.std(best_acc_list)
        print(f"source: {source}, mean_acc: {mean_acc:.2f}, std_acc: {std_acc:.2f}")

def create_dir_if_not_exists(path):
    # 构建完整的路径
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    main(args)

