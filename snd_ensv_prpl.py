# -*- encoding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
from models import PRPL
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset

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

def load_seed(args):

    if args.dset == 'seed':
        EEG, Label, Group = SEEDFeatureDataset(args.path, session=1).data()
        Label += 1
    if args.dset == 'seediv':
        EEG, Label, Group = SEEDIVFeatureDataset(args.path, session=2).data() #SEEDIV的Label不用加1
    EEG = EEG.reshape(-1, 310)
    sGroup = Group[:, 1]
    

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(sGroup):
        EEG[sGroup==i] = min_max_scaler.fit_transform(EEG[sGroup == i])

    one_hot_mat = np.eye(len(Label), args.class_num)[Label].astype("float32") #seediv有4类
    
    # 获得目标域数据
    target_features = torch.from_numpy(EEG[sGroup==args.target]).type(torch.Tensor)
    target_labels = torch.from_numpy(one_hot_mat[sGroup==args.target])
    torch_dataset_target = CustomDataset(target_features, target_labels)

    return torch_dataset_target

def entropy(p, mean=True):
    p = F.softmax(p, dim=1)
    if not mean:
        return -torch.sum(p * torch.log(p+1e-5), 1)
    else:
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))
    
def ent(p, mean=True):#
    if not mean:
        return -torch.sum(p * torch.log(p+1e-5), 1)
    else:
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))
  
def score(args, model, feature, labels):
    """
    计算模型的性能指标。
    
    参数:
        args: 配置参数。
        model: 已加载的模型。
        feature: 目标域的特征数据。
        labels: 目标域的标签数据。
    
    返回:
        acc: 准确率。
        ent_soft: 软预测的熵。SND
        ent_class: 类别预测的熵。Entropy
        all_output: 模型的预测输出。logits
    """
    model.eval()
    with torch.no_grad():
        # 预测
        y_preds = model.predict(feature)
        # print(y_preds)
        # print(y_preds.shape)
        # print(labels.shape)
        acc = np.sum(y_preds == labels) / len(labels)
        # print(acc)

        # 计算软预测的熵
        all_output = model.predict_prob(feature)  # logits
        # print(all_output)
        normalized = F.normalize(all_output/2.0, dim=1).cpu()  # 正则化
        s_mat = torch.matmul(normalized, normalized.t()) 
        tau = 0.1*torch.std(s_mat)
        mat = s_mat / tau
        mask = torch.eye(mat.size(0), mat.size(0)).bool()  # 对角线掩码
        mat.masked_fill_(mask, -1 / 0.05)  # 将对角线设为负无穷
        ent_soft = entropy(mat) # SND

        # 计算类别预测的熵
        ent_class = ent(all_output) # Entropy

        all_sfmx = nn.Softmax(dim=-1)(all_output)  # softmax
        mean_sfmx = torch.mean(all_sfmx, dim=0)  # 平均softmax
        # im: max is best
        mean_div = -torch.sum(mean_sfmx * torch.log(mean_sfmx + 1e-5)).item()
        mean_im = mean_div - ent_class.item()

        # corr-c: min is best
        ori_corr = torch.mm(all_sfmx.t(), all_sfmx)
        sfmxcorr = ori_corr.diag().sum().item() / ((ori_corr**2).sum()**0.5).item()
        
    return acc, ent_soft, ent_class, all_output, mean_im, sfmxcorr

def val(args, model_list, is_five=True):
    setup_seed(args.seed)
    # 提前加载目标域数据
    dataset_target = load_seed(args)
    loader_target = DataLoader(
        dataset=dataset_target,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    feature = loader_target.dataset.data()
    labels = loader_target.dataset.label()  # one-hot
    labels = np.argmax(labels.numpy(), axis=1)  # 转换为类别标签

    # 提前加载所有模型到内存
    loaded_models = {}
    for model_path in model_list:
        model = PRPL(num_of_class=args.class_num)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        loaded_models[model_path] = model

    # 缓存每个模型的预测结果
    model_results = {}
    def compute_score(model_path):
        model = loaded_models[model_path]  # 使用已加载的模型
        return model_path, score(args, model, feature, labels)  # 调用 score 函数

    # 使用多线程并行计算每个模型的 score
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_score, model_list))

    # 将结果存储到 model_results 中
    model_results = {model_path: result for model_path, result in results}

    print(f"Target{args.target} start model selection----------------")
    flu_level = args.flu_level  # 设置波动系数
    iteration = 0  # 初始化迭代次数
    while True:  # 持续循环，直到不需要排除为止
        score_list = []
        ent_list = []
        im_list = []
        corrc_list = []
        _, _, _, p, _, _ = model_results[model_list[0]]  # 获取第一个模型的预测结果以初始化 preds
        preds = torch.zeros((p.size()[0], args.class_num), device='cpu')

        # 使用缓存的预测结果
        for model_path in model_list:
            acc, ent_soft, ent_class, pred, im, corrc = model_results[model_path]
            score_list.append(ent_soft)
            ent_list.append(ent_class)
            im_list.append(im)
            corrc_list.append(corrc)
            preds += nn.Softmax(dim=-1)(pred)
        score_list = torch.tensor(score_list)
        ent_list = torch.tensor(ent_list)
        im_list = torch.tensor(im_list)
        corrc_list = torch.tensor(corrc_list)

        # 计算集成预测的准确率
        ac_list = []
        preds /= len(model_list)  # 平均预测结果
        ensem_pl = np.argmax(preds.cpu().numpy(), axis=1)  # 集成预测结果
        # print("-------------------------------------------------")
        for model_path in model_list:
            a, snd, entro, pred, _, _ = model_results[model_path]
            predict_label = np.argmax(pred.cpu().numpy(), axis=1)
            ac = np.sum(predict_label == ensem_pl) / len(ensem_pl)
            ac_list.append(ac)
            # print(f"snd:{snd}, entropy:{entro}, ensemble:{ac}")
        ac_list = torch.tensor(ac_list)

        def normolized_scores(score_list):
            # 归一化分数
            max_score = torch.max(score_list)
            min_score = torch.min(score_list)
            normolized_score = (score_list - min_score) / (max_score - min_score + 1e-5)
            return normolized_score
        
        score_list_norm = normolized_scores(score_list)  # SND
        ent_list_norm = normolized_scores(ent_list)  # Entropy
        ac_list_norm = normolized_scores(ac_list)  # Ensemble
        SND_WEIGHT = 0.5 # SND的权重
        ENT_WEIGHT = 0.5 #Entropy的权重
        ENS_WEIGHT = 1-SND_WEIGHT-ENT_WEIGHT # Ensemble的权重
        final_list = SND_WEIGHT * score_list_norm + ENS_WEIGHT * ac_list_norm - ENT_WEIGHT * ent_list_norm # 加权平均
        flu_level = args.flu_level * (1 + 0.1 * iteration)  # 动态调整波动系数,逐步放宽
        worst_model_idx = torch.argmin(ac_list)
        best_score_idx = torch.argmax(score_list)
        best_ent_idx = torch.argmin(ent_list)
        best_final_idx = torch.argmax(final_list)
        # 检查是否需要排除最差模型
        if max(ac_list) - min(ac_list) >= flu_level:
            # 特殊情况不排除
            if args.dset == 'seed' and is_five == True:
                if iteration > 0 and ((worst_model_idx == best_score_idx or worst_model_idx == best_ent_idx) and \
                    max(ac_list) < 1 and (max(ac_list)/2.0) > min(ac_list) or ((max(ac_list)/2.0) > min(ac_list) and max(ent_list) < 1)):
                    print(f"Excluded worst model. Remaining models: {len(model_list)}")
                    best_model = model_list[worst_model_idx]
                    best_model_weight = best_model.split("_weight")[1].split("\\")[0]  # 提取模型名称,zhendui多对一的情况
                    best_acc = model_results[best_model][0]
                    print(f"acc: {best_acc*100:.2f}")
                    print(f"After excluding, snd and entropy target{args.target}'s best model is : {best_model_weight}")
                    break  # 退出循环
            # 排除最差模型
            worst_model = model_list[worst_model_idx]
            if is_five == True:
                worst_model = worst_model.split("_weight")[1].split("\\")[0]  # 提取最差模型名称，针对多对一的情况
            else:
                worst_model = worst_model.split("_to")[0].split("\\")[-1]  # 提取最差模型名称，针对一对一的情况
            print(f"ensv target{args.target}'s worst model is: {worst_model}")
            model_list.pop(worst_model_idx)  # 从模型列表中移除最差模型
        else:
            # 不需要再排除，选择最佳模型并退出循环
            print(f"Excluded worst model. Remaining models: {len(model_list)}")
            print(f"left models: ", end="")
            for i in range(len(model_list)):
                if is_five == True:
                    left_model = model_list[i].split("_weight")[1].split("\\")[0]  # 提取模型名称,zhendui多对一的情况
                else:
                    left_model = model_list[i].split("_to")[0].split("\\")[-1]  # 提取模型名称,针对一对一的情况
                print(f"{left_model} ", end="")
            print()
            best_model = model_list[torch.argmax(final_list)]
            # best_model = model_list[torch.argmax(ac_list)] #消融，此时只有EnsV
            # best_model = model_list[torch.argmin(ent_list)] #entropy
            # best_model = model_list[torch.argmax(im_list)] #im
            # best_model = model_list[torch.argmin(corrc_list)] #corrc
            if is_five == True:
                best_model_weight = best_model.split("_weight")[1].split("\\")[0]  # 提取模型名称,zhendui多对一的情况
            else:
                best_model_weight = best_model.split("_to")[0].split("\\")[-1]  # 提取模型名称,针对一对一的情况
            best_acc = model_results[best_model][0]
            print(f"acc: {best_acc*100:.2f}")
            print(f"After excluding, snd and entropy target{args.target}'s best model is : {best_model_weight}")
            break  # 退出循环
        iteration += 1  # 增加迭代次数
        
def single_val(args):# 计算一次，排除一次
    
    # 构建模型路径列表
    model_list = []
    if args.dset == 'seed':
        for source in range(1, 16):  # 每个目标域14个源域分别训练
            if args.target == source:
                continue
            model_list.append(f".\\logs\\logs_prpl_dann\\session_1\\source{source}_to_target{args.target}\\best.pth")
    else:
        for source in range(1, 16):  # 每个目标域14个源域分别训练
            if args.target == source:
                continue
            model_list.append(f".\\logs\\logs_prpl_dann_iv\\session_2\\source{source}_to_target{args.target}\\best.pth")
        
    val(args, model_list, False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation for unsupervised learning')

    # parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dset', type=str, default='seed', choices=['seed', 'seediv'], help="dataset") #记得换
    parser.add_argument('--seed', type=int, default=20, help="seed")
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--target', type=int, default=1, help="定义对哪个目标域进行模型选择,从1开始")
    # SEED-IV best is 0.15
    parser.add_argument('--flu_level', type=float, default=100, help="定义波动系数，较大时需要排除一些模型")
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.dset == 'seed':
        args.path = "D:\\Desktop\\research\\SEED\\ExtractedFeatures\\"
        args.class_num = 3
        model_list = [f".\\logs\\logs_prpl_dann_weight0.01\\session_1\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_weight0.1\\session_1\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_weight0.5\\session_1\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_weight1\\session_1\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_weight2\\session_1\\{args.target}\\best.pth"]
    
    if args.dset == 'seediv':
        args.path = "D:\\Desktop\\research\\SEED_IV\\eeg_feature_smooth\\"
        args.class_num = 4
        model_list = [f".\\logs\\logs_prpl_dann_iv_weight0.01\\session_2\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_iv_weight0.1\\session_2\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_iv_weight0.5\\session_2\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_iv_weight1\\session_2\\{args.target}\\best.pth",
                  f".\\logs\\logs_prpl_dann_iv_weight2\\session_2\\{args.target}\\best.pth"]
    
    # 五个候选模型，一个目标域对一个源域，一个源域有14个受试者
    val(args, model_list)

    # 14个候选模型，一个目标域对14个源域，一个源域是一个受试者
    # single_val(args)
