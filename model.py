import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, RGCNConv
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_scatter import segment_csr
from scipy.sparse import coo_matrix
import numpy as np
import pickle

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def CalculateSimilarity(emb):
    similarity = emb / (torch.norm(emb, dim=-1, keepdim=True) + 1e-10)
    similarity = torch.mm(similarity, similarity.T)
    return similarity


def torch_corr_x_y(tensor1, tensor2):
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
    lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))
    lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))
    std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
    corr_x_y = torch.div(lxy, std_x_y)
    return corr_x_y


def scale_sigmoid(tensor, alpha):
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    output = torch.sigmoid(torch.mul(alpha, tensor))
    return output


# Obtain the topk neighbors for each cell line
class SimGraphConstruction(nn.Module):
    def __init__(self, k, device):
        super(SimGraphConstruction, self).__init__()
        self.k = k
        self.device = device

    def forward(self, feature):
        sim = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-10)
        sim = torch.mm(sim, sim.T)

        diag = torch.diag(sim)
        diag = torch.diag_embed(diag)
        sim = sim - diag

        tmp = torch.topk(sim, dim=1, k=self.k)
        index = torch.arange(tmp.values.shape[0]).unsqueeze(1).to(self.device)
        edge = torch.empty((tmp.values.shape[0] * self.k, 2), dtype=int)
        for i in range(self.k):
            index_tmp = torch.cat((index, tmp.indices[:, i].unsqueeze(1)), 1)
            edge[tmp.values.shape[0] * i:tmp.values.shape[0] * (i + 1), :] = index_tmp
        edge = edge.T.to(self.device)

        edge_weights = tmp.values.t().reshape(-1).to(self.device)

        return edge, edge_weights


# Multi-omics encoder and drug encoder in the attribute branch
class AttributeBranch(nn.Module):
    def __init__(self, dim_drug, drug_layer, dim_gexp, dim_methy, dim_feat, k, num_layers, dropout, use_bn_at, negative_slope, device):
        super(AttributeBranch, self).__init__()
        self.device = device
        self.dropout = dropout
        self.leakyRelu = nn.LeakyReLU(negative_slope)
        self.use_bn_at = use_bn_at

        # Drug encoder
        self.drug_layer = drug_layer
        self.drug_conv = GCNConv(dim_drug, drug_layer[0])
        self.drug_graph_bn1 = nn.BatchNorm1d(drug_layer[0])
        self.graph_conv = []
        self.graph_bn = []
        for i in range(len(drug_layer) - 1):
            self.graph_conv.append(GCNConv(drug_layer[i], drug_layer[i + 1]).to(device))
            self.graph_bn.append(nn.BatchNorm1d((drug_layer[i + 1])).to(device))
        self.conv_end = GCNConv(drug_layer[-1], dim_feat)
        self.batch_end = nn.BatchNorm1d(dim_feat)

        # Multi-omics encoder
        self.k = k
        self.sim_graph = SimGraphConstruction(k, device)
        self.num_layers = num_layers

        # Transform three raw omics data into latent representations with unified dimension dim_feat
        self.gexp_fc1 = nn.Linear(dim_gexp, 256)
        self.gexp_fc2 = nn.Linear(256, dim_feat)
        self.batch_gexp = nn.BatchNorm1d(256)

        self.methy_fc1 = nn.Linear(dim_methy, 256)
        self.methy_fc2 = nn.Linear(256, dim_feat)
        self.batch_methy = nn.BatchNorm1d(256)

        self.mut_cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.mut_cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.mut_fla = nn.Flatten()
        self.mut_fc = nn.Linear(2010, dim_feat)
        self.batch_mut1 = nn.BatchNorm2d(50)
        self.batch_mut2 = nn.BatchNorm2d(30)
        self.batch_mut3 = nn.BatchNorm1d(2010)

        # Omics-specific representation learning
        self.gexp_sim_graph = []
        self.methy_sim_graph = []
        self.mut_sim_graph = []
        self.gexp_sim_graph.append(GCNConv(dim_feat*3, dim_feat).to(device))
        self.methy_sim_graph.append(GCNConv(dim_feat*3, dim_feat).to(device))
        self.mut_sim_graph.append(GCNConv(dim_feat*3, dim_feat).to(device))
        for i in range(num_layers - 1):
            self.gexp_sim_graph.append(GCNConv(dim_feat, dim_feat).to(device))
            self.methy_sim_graph.append(GCNConv(dim_feat, dim_feat).to(device))
            self.mut_sim_graph.append(GCNConv(dim_feat, dim_feat).to(device))

        self.cell_base = nn.Linear(dim_feat*3, dim_feat*1)
        self.cell_fc = nn.Linear(dim_feat, dim_feat)

        # Final representation
        self.batch_all = nn.BatchNorm1d(dim_feat)

    def forward(self, drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data):
        # Drug encoder
        drug_adj = SparseTensor(row=drug_adj[0], col=drug_adj[1]).t()
        x_drug = self.drug_conv(drug_feature, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.drug_graph_bn1(x_drug)
        for i in range(len(self.drug_layer) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)
        x_drug_all = segment_csr(x_drug, drug_batch, reduce='max')


        # Multi-omics Encoder
        # Mutation representation
        x_mutation = torch.tanh(self.mut_cov1(mutation_data))
        x_mutation = self.batch_mut1(x_mutation)
        x_mutation = F.max_pool2d(x_mutation, (1, 5))
        x_mutation = F.relu(self.mut_cov2(x_mutation))
        x_mutation = self.batch_mut2(x_mutation)
        x_mutation = F.max_pool2d(x_mutation, (1, 10))
        x_mutation = self.mut_fla(x_mutation)
        x_mutation = self.batch_mut3(x_mutation)
        x_mutation = F.relu(self.mut_fc(x_mutation))


        # Gene expression representation
        x_gexpr = torch.tanh(self.gexp_fc1(gexpr_data))
        x_gexpr = self.batch_gexp(x_gexpr)
        x_gexpr = F.relu(self.gexp_fc2(x_gexpr))

        # Methylation representation
        x_methylation = torch.tanh(self.methy_fc1(methylation_data))
        x_methylation = self.batch_methy(x_methylation)
        x_methylation = F.relu(self.methy_fc2(x_methylation))

        # The basic multi-omics representation by concatenation
        x_cell_base = self.cell_base(torch.cat((x_mutation, x_gexpr, x_methylation), 1))
        x_cell_base = self.leakyRelu(x_cell_base)
        x_cell_base = F.normalize(x_cell_base, dim=1)

        # Similarity graph construction for three omics
        mutation_sim_edge, _ = self.sim_graph(x_mutation)
        mutation_sim_edge = SparseTensor(row=mutation_sim_edge[0], col=mutation_sim_edge[1], sparse_sizes=(mutation_data.shape[0], mutation_data.shape[0]))

        gexpr_sim_edge, _ = self.sim_graph(x_gexpr)
        gexpr_sim_edge = SparseTensor(row=gexpr_sim_edge[0], col=gexpr_sim_edge[1], sparse_sizes=(gexpr_data.shape[0], gexpr_data.shape[0]))

        methylation_sim_edge, _ = self.sim_graph(x_methylation)
        methylation_sim_edge = SparseTensor(row=methylation_sim_edge[0], col=methylation_sim_edge[1], sparse_sizes=(methylation_data.shape[0], methylation_data.shape[0]))

        # Omics-specific representation learning
        x_mutation2 = self.mut_sim_graph[0](torch.cat((x_mutation, x_gexpr, x_methylation), 1), mutation_sim_edge)
        x_mutation2 = self.leakyRelu(x_mutation2)
        x_mutation2 = F.dropout(x_mutation2, self.dropout, training=self.training)
        for i in range(1, self.num_layers):
            x_mutation2 = self.mut_sim_graph[i](x_mutation2, mutation_sim_edge)
            x_mutation2 = self.leakyRelu(x_mutation2)
            x_mutation2 = F.dropout(x_mutation2, self.dropout, training=self.training)
        x_mutation2 = F.normalize(x_mutation2, dim=1)

        x_gexpr2 = self.gexp_sim_graph[0](torch.cat((x_mutation, x_gexpr, x_methylation), 1), gexpr_sim_edge)
        x_gexpr2 = self.leakyRelu(x_gexpr2)
        x_gexpr2 = F.dropout(x_gexpr2, self.dropout, training=self.training)
        for i in range(1, self.num_layers):
            x_gexpr2 = self.gexp_sim_graph[i](x_gexpr2, gexpr_sim_edge)
            x_gexpr2 = self.leakyRelu(x_gexpr2)
            x_gexpr2 = F.dropout(x_gexpr2, self.dropout, training=self.training)
        x_gexpr2 = F.normalize(x_gexpr2, dim=1)

        x_methylation2 = self.methy_sim_graph[0](torch.cat((x_mutation, x_gexpr, x_methylation), 1), methylation_sim_edge)
        x_methylation2 = self.leakyRelu(x_methylation2)
        x_methylation2 = F.dropout(x_methylation2, self.dropout, training=self.training)
        for i in range(1, self.num_layers):
            x_methylation2 = self.methy_sim_graph[i](x_methylation2, methylation_sim_edge)
            x_methylation2 = self.leakyRelu(x_methylation2)
            x_methylation2 = F.dropout(x_methylation2, self.dropout, training=self.training)
        x_methylation2 = F.normalize(x_methylation2, dim=1)

        # Dot-product attention
        query = x_cell_base.unsqueeze(1)
        key = torch.cat((x_mutation2.unsqueeze(1), x_gexpr2.unsqueeze(1), x_methylation2.unsqueeze(1)), 1)
        cos_sim = nn.Softmax(dim=2)(torch.matmul(query, key.permute(0, 2, 1))).squeeze(1)


        x_cell = (x_mutation2) * cos_sim[:, 0].reshape(-1, 1) + (x_gexpr2) * cos_sim[:, 1].reshape(-1, 1) + (x_methylation2) * cos_sim[:, 2].reshape(-1, 1)
        x_cell_all = F.relu(self.cell_fc(x_cell))

        # Final representation for cell lines and drugs in the attribute branch
        x_all = torch.cat((x_cell_all, x_drug_all), 0)
        if self.use_bn_at:
            x_all = self.batch_all(x_all)

        return x_all


# Representation learning in the association branch
class AssociationBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, num_cell, num_drug, use_bn_as, negative_slope, device):
        super(AssociationBranch, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_cell = num_cell
        self.num_drug = num_drug
        self.leakyRelu = nn.LeakyReLU(negative_slope)
        self.use_bn_as = use_bn_as

        # Sensitive graph
        self.sen_conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.sen_conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.sen_fc = nn.Linear(hidden_channels * 2 + in_channels, hidden_channels)
        self.sen_bn = nn.BatchNorm1d(hidden_channels)

        # Resistant graph
        self.res_conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.res_conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.res_fc = nn.Linear(hidden_channels * 2 + in_channels, hidden_channels)
        self.res_bn = nn.BatchNorm1d(hidden_channels)

        # Fusion of dual graph
        self.cell_fc = nn.Linear(hidden_channels*2, hidden_channels)
        self.drug_fc = nn.Linear(hidden_channels*2, hidden_channels)


    def forward(self, feature, edge):

        # Obtain the sensitive and resistant edge
        sen_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2]).T.long().to(self.device)
        res_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2]).T.long().to(self.device)

        sen_edge = SparseTensor(row=sen_edge[0], col=sen_edge[1], sparse_sizes=(self.num_cell + self.num_drug, self.num_cell + self.num_drug)).t()
        res_edge = SparseTensor(row=res_edge[0], col=res_edge[1], sparse_sizes=(self.num_cell + self.num_drug, self.num_cell + self.num_drug)).t()

        # Sensitive graph representation
        x_sen = self.sen_conv1(feature, sen_edge)
        x_sen = self.leakyRelu(x_sen)
        x_sen = F.dropout(x_sen, self.dropout, training=self.training)
        x_sen2 = self.sen_conv2(x_sen, sen_edge)
        x_sen2 = self.leakyRelu(x_sen2)
        x_sen2 = F.dropout(x_sen2, self.dropout, training=self.training)

        x_sen_all = torch.cat((feature, x_sen, x_sen2), 1)
        x_sen_all = self.sen_fc(x_sen_all)
        x_sen_all = F.relu(x_sen_all)
        if self.use_bn_as:
            x_sen_all = self.sen_bn(x_sen_all)

        # Resistant graph representation
        x_res = self.res_conv1(feature, res_edge)
        x_res = self.leakyRelu(x_res)
        x_res = F.dropout(x_res, self.dropout, training=self.training)
        x_res2 = self.res_conv2(x_res, res_edge)
        x_res2 = self.leakyRelu(x_res2)
        x_res2 = F.dropout(x_res2, self.dropout, training=self.training)

        x_res_all = torch.cat((feature, x_res, x_res2), 1)
        x_res_all = self.res_fc(x_res_all)
        x_res_all = F.relu(x_res_all)
        if self.use_bn_as:
            x_res_all = self.res_bn(x_res_all)

        # Final representations of cell lines and drugs in the association branch
        x_sen_cell = x_sen_all[:self.num_cell, ]
        x_sen_drug = x_sen_all[self.num_cell:, ]
        x_res_cell = x_res_all[:self.num_cell, ]
        x_res_drug = x_res_all[self.num_cell:, ]

        x_cell = self.cell_fc(torch.cat((x_sen_cell, x_res_cell), 1))
        x_drug = self.drug_fc(torch.cat((x_sen_drug, x_res_drug), 1))
        x_cell = nn.ReLU()(x_cell)
        x_drug = nn.ReLU()(x_drug)

        return x_cell, x_drug


class RedCDR(nn.Module):
    def __init__(self, dim_drug, drug_layer, dim_gexp, dim_methy, dim_feat, dim_out, num_cell, num_drug, k, num_layers, dropout,
                 alpha, use_bn_at, use_bn_as, use_bn_fu, negative_slope, device):
        super(RedCDR, self).__init__()
        self.device = device
        self.num_cell = num_cell
        self.num_drug = num_drug
        self.alpha = alpha
        self.Relu = nn.ReLU()
        self.use_bn_fu = use_bn_fu

        self.branch_at = AttributeBranch(dim_drug, drug_layer, dim_gexp, dim_methy, dim_feat, k, num_layers, dropout, use_bn_at, negative_slope, device)
        self.branch_as = AssociationBranch(dim_feat, dim_out, dropout, num_cell, num_drug, use_bn_as, negative_slope, device)

        self.dual_cell_fc = nn.Linear(dim_out*2, dim_out)
        self.dual_drug_fc = nn.Linear(dim_out*2, dim_out)

        self.final_cell_fc = nn.Linear(dim_feat + dim_out, dim_out)
        self.final_drug_fc = nn.Linear(dim_feat + dim_out, dim_out)

        self.C_emb = nn.Parameter(torch.rand(num_cell, dim_feat), requires_grad=True)
        self.D_emb = nn.Parameter(torch.rand(num_drug, dim_feat), requires_grad=True)

        self.bn_final_c = nn.BatchNorm1d(dim_feat)
        self.bn_final_d = nn.BatchNorm1d(dim_feat)


    def forward(self, drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data, edge):
        # The representation of cell lines and drugs in the attribute branch
        feat_at = self.branch_at(drug_feature, drug_adj, drug_batch, mutation_data, gexpr_data, methylation_data)
        feat_at_cell = feat_at[:self.num_cell, ]
        feat_at_drug = feat_at[self.num_cell:, ]

        # The representation of cell lines and drugs in the association branch
        initial_feature = torch.cat((self.C_emb, self.D_emb), 0)
        feat_as_cell, feat_as_drug = self.branch_as(initial_feature, edge)

        # Fusion of two branches
        final_cell = torch.cat((feat_at_cell, feat_as_cell), 1)
        final_drug = torch.cat((feat_at_drug, feat_as_drug), 1)
        final_cell = self.final_cell_fc(final_cell)
        final_drug = self.final_drug_fc(final_drug)
        final_cell = self.Relu(final_cell)
        final_drug = self.Relu(final_drug)
        if self.use_bn_fu:
            final_cell = self.bn_final_c(final_cell)
            final_drug = self.bn_final_d(final_drug)

        # Prediction probability of fusion module
        corr = torch_corr_x_y(final_cell, final_drug)
        final_prob = scale_sigmoid(corr, alpha=self.alpha)

        # Prediction probability of each branch
        corr_at = torch_corr_x_y(feat_at_cell, feat_at_drug)
        corr_as = torch_corr_x_y(feat_as_cell, feat_as_drug)
        at_prob = scale_sigmoid(corr_at, alpha=self.alpha)
        as_prob = scale_sigmoid(corr_as, alpha=self.alpha)

        # Representation distillation
        cell_similarity_at = CalculateSimilarity(feat_at_cell)
        drug_similarity_at = CalculateSimilarity(feat_at_drug)

        cell_similarity_as = CalculateSimilarity(feat_as_cell)
        drug_similarity_as = CalculateSimilarity(feat_as_drug)

        cell_similarity_fusion = CalculateSimilarity(final_cell)
        drug_similarity_fusion = CalculateSimilarity(final_drug)

        loss_mse = torch.nn.MSELoss().to(self.device)
        rd_loss = loss_mse(cell_similarity_at, cell_similarity_fusion) + loss_mse(drug_similarity_at, drug_similarity_fusion) \
                        + loss_mse(cell_similarity_as, cell_similarity_fusion) + loss_mse(drug_similarity_as, drug_similarity_fusion)

        # Prediction distillation
        pd_loss = loss_mse(final_prob.view(-1), at_prob.view(-1)) + loss_mse(final_prob.view(-1), as_prob.view(-1))

        return final_prob.view(-1), rd_loss, pd_loss
