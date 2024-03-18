import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import pickle

from myutils import GraphDataset, collate
from scipy.sparse import coo_matrix


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list, _ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


# Tranform the data into inputs
def process(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs):
    # construct cell line-drug response pairs
    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    pubmedid = list(set([item[1] for item in data_new]))
    pubmedid.sort()
    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))
    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])
    pubmed_num = np.squeeze([[j[1] for j in pubmedmap if i[1] == j[0]] for i in data_new])
    IC_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((cellline_num, pubmed_num, IC_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    # process drug feature
    pubid = [item[0] for item in pubmedmap]
    drug_feature = pd.DataFrame(drug_feature).T
    drug_feature = drug_feature.loc[pubid]
    atom_shape = drug_feature[0][0].shape[-1]
    drug_data = FeatureExtract(drug_feature)

    #----cell line_feature_input
    cellid = [item[0] for item in cellmap]
    gexpr_feature = gexpr_feature.loc[cellid]
    mutation_feature = mutation_feature.loc[cellid]
    methylation_feature = methylation_feature.loc[cellid]

    mutation = torch.from_numpy(np.array(mutation_feature, dtype='float32'))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature, dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation_feature, dtype='float32'))

    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data), collate_fn=collate, batch_size=nb_drugs, shuffle=False, num_workers=0)
    cellline_set = Data.DataLoader(dataset=Data.TensorDataset(mutation, gexpr, methylation), batch_size=nb_celllines, shuffle=False, num_workers=0)

    return drug_set, cellline_set, allpairs, atom_shape


def cmask(num, train_ratio, valid_ratio, seed):
    mask = np.zeros(num)
    mask[0:int(train_ratio * num)] = 0
    mask[int(train_ratio * num):int((train_ratio+valid_ratio) * num)] = 1
    mask[int((train_ratio + valid_ratio) * num):] = 2
    np.random.seed(seed)
    np.random.shuffle(mask)
    train_mask = (mask == 0)
    valid_mask = (mask == 1)
    test_mask = (mask == 2)
    return train_mask, valid_mask, test_mask


# Split the response into train/valid/test set
def process_label_random(allpairs, nb_celllines, nb_drugs, train_ratio=0.6, valid_ratio=0.2, seed=100):
    # split into positive and negative pairs
    pos_pairs = allpairs[allpairs[:, 2] == 1]
    neg_pairs = allpairs[allpairs[:, 2] == -1]
    pos_num = len(pos_pairs)
    neg_num = len(neg_pairs)

    # random
    train_mask, valid_mask, test_mask = cmask(len(allpairs), train_ratio, valid_ratio, seed)
    train = allpairs[train_mask][:, 0:3]
    valid = allpairs[valid_mask][:, 0:3]
    test = allpairs[test_mask][:, 0:3]

    train_edge = np.vstack((train, train[:, [1, 0, 2]]))
    train[:, 1] -= nb_celllines
    test[:, 1] -= nb_celllines
    valid[:, 1] -= nb_celllines

    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    valid_mask = coo_matrix((np.ones(valid.shape[0], dtype=bool), (valid[:, 0], valid[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()

    pos_pairs[:, 1] -= nb_celllines
    neg_pairs[:, 1] -= nb_celllines
    label_pos = coo_matrix((np.ones(pos_pairs.shape[0]), (pos_pairs[:, 0], pos_pairs[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)

    return train_mask, valid_mask, test_mask, train_edge, label_pos





