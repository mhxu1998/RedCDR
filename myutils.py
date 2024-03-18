from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
from sklearn import metrics

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features, edge_index = data_mol[0], data_mol[1]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA



def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    with torch.no_grad():
        thresholds = torch.unique(predict_data)
    size = torch.tensor([thresholds.size()[0], true_data.size()[0]], dtype=torch.int32, device=true_data.device)
    ones = torch.ones([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1)
    tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
    two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
    n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
    scores = torch.div(torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn)))
    max_f1_score = torch.max(scores)
    threshold = thresholds[torch.argmax(scores)]
    return max_f1_score, threshold


def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    n = true_data.size()[0]
    ones = torch.ones(n, dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(n, dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
    score = torch.div(tpn, n)
    return score


def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    true_neg = torch.sub(ones, true_data)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score


def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data))
    score = torch.div(tp, torch.add(tp, fn))
    return score


def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data)
    tp = torch.sum(torch.mul(true_data, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
    score = torch.div((tp * tn - fp * fn), torch.add(delta, torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    return score



def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt.detach().cpu().numpy(), yp.detach().cpu().numpy())
    aupr = metrics.auc(recall, precision)
    # aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy())
    #---f1,acc,recall, specificity, precision
    f1, thresholds = f1_score_binary(yt, yp)
    acc = accuracy_binary(yt, yp, thresholds)
    precision = precision_binary(yt, yp, thresholds)
    recall = recall_binary(yt, yp, thresholds)
    mcc = mcc_binary(yt, yp, thresholds)

    return auc, aupr, f1.detach().cpu().numpy(), acc.detach().cpu().numpy(), precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), mcc.detach().cpu().numpy()
