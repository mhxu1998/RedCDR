import numpy as np
import csv
import pandas as pd
import os
import pickle

from rdkit import Chem
import deepchem as dc
import pubchempy as pcp


def LoadData(Drug_info_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file,
              Gene_expression_file, Methylation_file, Cancer_response_exp_file, IC50_threds_file):
    print('Loading data...')
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]

    print('1-loading drugs information')
    # [drugid->pubchemid]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # load drug threshold [pubchemid->thred]
    drug2thred = {}
    for line in open(IC50_threds_file).readlines()[1:]:
        drug2thred[str(line.split('\t')[0])] = float(line.strip().split('\t')[1])

    # load drug molecular graph feature [pubchemid->feature]
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        pkl_file = open('%s/%s' % (Drug_feature_file, each), 'rb')
        feat_mat, adj_list, degree_list, func = pickle.load(pkl_file)  # func is not used
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    print('2-loading cell lines information')
    # load cell line cancer type  [ccleid->cancertype]
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    # load cell line omics information
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]

    print('3-loading response information')
    # load cell line-drug response information
    response_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    drug_match_list = [item for item in response_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    response_data_filtered = response_data.loc[drug_match_list]
    data_idx = []
    use_thred = True
    for each_drug in response_data_filtered.index:
        for each_cellline in response_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            # Select drugs and cell lines with feature information
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(response_data_filtered.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(response_data_filtered.loc[each_drug, each_cellline])
                    if use_thred:
                        if pubchem_id in drug2thred.keys():
                            binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else -1
                            data_idx.append((each_cellline, pubchem_id, binary_IC50, cellline2cancertype[each_cellline]))
                    else:
                        binary_IC50 = 1 if ln_IC50 < -2 else -1
                        data_idx.append((each_cellline, pubchem_id, binary_IC50, cellline2cancertype[each_cellline]))

    # eliminate ambiguity responses
    data_sort = sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
    data_tmp = []
    data_new = []
    data_idx1 = [[i[0], i[1]] for i in data_sort]
    for i, k in zip(data_idx1, data_sort):
        if i not in data_tmp:
            data_tmp.append(i)
            data_new.append(k)
    nb_celllines = len(set([item[0] for item in data_new]))
    nb_drugs = len(set([item[1] for item in data_new]))
    print('All %d pairs across %d cell lines and %d drugs.'%(len(data_new),nb_celllines,nb_drugs))
    return drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs


def LoadData_ccle(Drug_feature_file, Genomic_mutation_file, Gene_expression_file, Methylation_file, Cancer_response_exp_file):
    print('Loading data...')

    print('1-loading drugs information')
    # load drug feature
    drug = pd.read_csv(Drug_feature_file, sep=',', header=0)
    drug_feature = {}
    featurizer = dc.feat.ConvMolFeaturizer()
    for tup in zip(drug['pubchem'], drug['isosmiles']):
        mol = Chem.MolFromSmiles(tup[1])
        X = featurizer.featurize(mol)
        drug_feature[str(tup[0])] = [X[0].get_atom_features(), X[0].get_adjacency_list(), 1]

    print('2-loading cell lines information')
    # load cell line omics information
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]

    print('3-loading response information')
    # load cell line-drug response information
    datar = pd.read_csv(Cancer_response_exp_file, sep=',', header=0)
    data_idx = []
    thred = 0.8
    for tup in zip(datar['DepMap_ID'], datar['pubchem'], datar['Z_SCORE']):
        t = 1 if tup[2] > thred else -1
        data_idx.append((tup[0], str(tup[1]), t))

    # eliminate ambiguity responses
    data_sort = sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
    data_tmp = []
    data_new = []
    data_idx1 = [[i[0], i[1]] for i in data_sort]
    for i, k in zip(data_idx1, data_sort):
        if i not in data_tmp:
            data_tmp.append(i)
            data_new.append(k)
    nb_celllines = len(set([item[0] for item in data_new]))
    nb_drugs = len(set([item[1] for item in data_new]))
    print('All %d pairs across %d cell lines and %d drugs.' % (len(data_new), nb_celllines, nb_drugs))

    return drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs

