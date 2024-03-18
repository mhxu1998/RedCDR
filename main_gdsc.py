import torch
import numpy as np
import random

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from load_data import *
from process_data import *
import pickle
import argparse
import time
from model import *
from myutils import metrics_graph
import pandas as pd
from tqdm import tqdm
from thop import profile


torch.set_num_threads(3)

parser = argparse.ArgumentParser(description="Run")
parser.add_argument('--device', type=str, default="cpu", help='cuda:number or cpu')
parser.add_argument('--valid_ratio', type=float, default=0.2, help="the ratio for valid set")
parser.add_argument('--test_ratio', type=float, default=0.2, help="the ratio for test set")
parser.add_argument('--outfile', type=str, default="RedCDR_gdsc", help="output file")
parser.add_argument('--save_model', type=bool, default=None, help="save the model checkpoint")

parser.add_argument('--seed', type=int, default=2023, help='random seed that divides the dataset')
parser.add_argument('--epochs', type=int, default=2000, help="the epochs for model")
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
parser.add_argument('--dropout', type=float, default=0.4, help="drop out rate")
parser.add_argument('--numk', type=int, default=5, help="knn sim graph")
parser.add_argument('--dim_feat', type=int, default=100, help="Dimension of the latent representation")
parser.add_argument('--layers', type=int, default=2, help="Layers of gnn in attribute branch")
parser.add_argument('--rd', type=float, default=0.5, help="Coefficient of representation distillation loss")
parser.add_argument('--pd', type=float, default=1.5, help="Coefficient of prediction distillation loss")
parser.add_argument('--alpha', type=float, default=8, help="The scaling parameter of sigmoid activation")

args = parser.parse_args()
start_time = time.time()

if args.device != "cpu":
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"


Drug_info_file = 'data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = 'data/Drug/drug_threshold.txt'
Drug_feature_file = 'data/Drug/new'
Cell_line_info_file = 'data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = 'data/Celline/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = 'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = 'data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
Cancer_response_exp_file = 'data/Celline/GDSC_IC50.csv'

drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = LoadData(Drug_info_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file, Gene_expression_file, Methylation_file, Cancer_response_exp_file, IC50_threds_file)
# Save the processed data so do not have to run it again the next time
# pickle.dump([drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs],open("data/gdsc.data","wb"))
# drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = pickle.load(open("data/gdsc.data","rb"))

drug_set, cellline_set, allpairs, atom_shape = process(drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs)


res = []
valid_res = []
ep = []
args_list = 'ep' + str(args.epochs) + '_' + 'lr' + str(args.lr) + '_' + 'drop' + str(args.dropout) + '_' + 'numk' \
            + str(args.numk) + '_' + 'rd' + str(args.rd) + '_' + 'pd' + str(args.pd) + '_' + 'dim' + str(args.dim_feat) + '_' + 'ratio' + str(args.test_ratio) + '_' + 'alpha' + str(args.alpha)
print(args_list)

if not os.path.exists("save"):
    os.mkdir("save")

# The experiment is repeated with 10 random seeds
for args.seed in range(2023, 2033):
    print('seed: '+ str(args.seed))
    train_mask, valid_mask, test_mask, train_edge, label_pos = process_label_random(allpairs, nb_celllines,
                                                                                    nb_drugs,
                                                                                    1-args.valid_ratio-args.test_ratio,
                                                                                    args.valid_ratio,
                                                                                    args.seed)

    train_mask = torch.from_numpy(train_mask).view(-1)
    valid_mask = torch.from_numpy(valid_mask).view(-1)
    test_mask = torch.from_numpy(test_mask).view(-1)
    
    model = RedCDR(atom_shape, [256,256,256], gexpr_feature.shape[-1], methylation_feature.shape[-1], args.dim_feat, 100, nb_celllines, nb_drugs, args.numk, args.layers, args.dropout, args.alpha, True, True, False, 0.2, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    bceloss = nn.BCELoss().to(device)
    label_pos = label_pos.to(device)

    best_AUC = 0; best_epoch = 0;
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0; final_P = 0; final_R = 0; final_M = 0

    for epoch in tqdm(range(args.epochs)):
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            drug.x, drug.edge_index, drug.batch = drug.x.to(device), drug.edge_index.to(device), drug.ptr.to(device)
            mutation_data, gexpr_data, methylation_data = cell[0].to(device), cell[1].to(device), cell[2].to(device)

            # train
            model.train()
            loss_temp = 0
            output, rd_loss, pd_loss = model(drug.x, drug.edge_index, drug.batch, mutation_data, gexpr_data, methylation_data, train_edge)
            optimizer.zero_grad()
            pos_loss = bceloss(output[train_mask], label_pos[train_mask])
            loss = pos_loss + args.rd*rd_loss + args.pd*pd_loss
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()

            if epoch % 200 == 0:
                print(pos_loss, rd_loss, pd_loss)

            # valid and test
            with torch.no_grad():
                model.eval()
                output, rd_loss, pd_loss = model(drug.x, drug.edge_index, drug.batch, mutation_data, gexpr_data, methylation_data, train_edge)
                pos_loss = bceloss(output[valid_mask], label_pos[valid_mask])
                vloss = pos_loss + args.rd*rd_loss + args.pd*pd_loss
                yvalid_p = output[valid_mask]
                yvalid_t = label_pos[valid_mask]
                valid_auc, valid_aupr, valid_f1, valid_acc, valid_precision, valid_recall, valid_mcc = metrics_graph(yvalid_t, yvalid_p)

                ytest_p = output[test_mask]
                ytest_t = label_pos[test_mask]
                test_auc, test_aupr, test_f1, test_acc, test_precision, test_recall, test_mcc = metrics_graph(ytest_t, ytest_p)

                if epoch % 200 == 0:
                    print('valid auc: ' + str(round(valid_auc, 4)) + '  valid aupr: ' + str(round(valid_aupr, 4)) +
                          '  valid f1: ' + str(np.round(valid_f1, 4)) + '  valid acc: ' + str(np.round(valid_acc, 4)) + '  valid p: ' + str(
                          np.round(valid_precision, 4)) + '  valid r: ' + str(np.round(valid_recall, 4)) + '  valid mcc: ' + str(np.round(valid_mcc, 4)))
                    print('test auc: ' + str(round(test_auc, 4)) + '  test aupr: ' + str(round(test_aupr, 4)) +
                          '  test f1: ' + str(np.round(test_f1, 4)) + '  test acc: ' + str(np.round(test_acc, 4)) + '  test p: ' + str(
                          np.round(test_precision, 4)) + '  test r: ' + str(np.round(test_recall, 4)) + '  test mcc: ' + str(np.round(test_mcc, 4)))

                if valid_auc > best_AUC:
                    best_AUC = valid_auc; best_epoch = epoch
                    final_AUC = test_auc; final_AUPR = test_aupr; final_F1 = test_f1; final_ACC = test_acc; final_P = test_precision; final_R = test_recall; final_M = test_mcc
                    if args.save_model:
                        torch.save(model.state_dict(), 'save/RedCDR_' + args_list + '.model' + str(args.seed))

    tmp = [args.seed, final_AUC, final_AUPR, final_F1, final_ACC, final_P, final_R, final_M]
    res.append(tmp)
    print(tmp)
    print(best_epoch)
    valid_res.append(best_AUC)
    ep.append(best_epoch)


if not os.path.exists("out"):
    os.mkdir("out")
output_path = "out/" + args.outfile + ".csv"
res_mean = np.array(res).mean(axis=0)
res_mean = list(res_mean)
res_mean[0] = 'mean'
res.append(res_mean)
res.append([np.mean(np.array(valid_res)), 'each: ', valid_res])
res.append([ep])
res.append(['  '])

df = pd.DataFrame(data=[args_list])
df.to_csv(output_path, header=False, index=False, mode='a')
df = pd.DataFrame(data=res)
df.to_csv(output_path, header=['seed', 'auc', 'aupr', 'f1', 'acc', 'prec', 'recall', 'mcc'], index=False, mode='a')








