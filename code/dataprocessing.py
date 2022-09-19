import csv
import torch
import random


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

def data_pro(args):
    dataset = dict()
    dataset['ld_p'] = read_csv(args.dataset_path + '/LD_adjmat.csv')
    dataset['ld_true'] = read_csv(args.dataset_path + '/LD_adjmat.csv')

    zero_index = []
    one_index = []
    for i in range(dataset['ld_p'].size(0)):
        for j in range(dataset['ld_p'].size(1)):
            if dataset['ld_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['ld_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)
    dataset['ld'] = dict()
    dataset['ld']['train'] = [one_tensor, zero_tensor]

    #disease semantic similarity
    dss_matrix = read_csv(args.dataset_path + '/DSS.csv')
    dss_edge_index = get_edge_index(dss_matrix)
    dataset['dss'] = {'data_matrix': dss_matrix, 'edges': dss_edge_index}

    #disease Gaussian interaction profile kernel similarity
    dgs_matrix = read_csv(args.dataset_path + '/DGS.csv')
    dgs_edge_index = get_edge_index(dgs_matrix)
    dataset['dgs'] = {'data_matrix': dgs_matrix, 'edges': dgs_edge_index}

    #disease cosine similarity
    dcs_matrix = read_csv(args.dataset_path + '/DCS.csv')
    dcs_edge_index = get_edge_index(dcs_matrix)
    dataset['dcs'] = {'data_matrix': dcs_matrix, 'edges': dcs_edge_index}


    #lncRNA functional similarity
    lfs_matrix = read_csv(args.dataset_path + '/LFS.csv')
    lfs_edge_index = get_edge_index(lfs_matrix)
    dataset['lfs'] = {'data_matrix': lfs_matrix, 'edges': lfs_edge_index}

    #lncRNA Gaussian interaction profile kernel similarity
    lgs_matrix = read_csv(args.dataset_path + '/LGS.csv')
    lgs_edge_index = get_edge_index(lgs_matrix)
    dataset['lgs'] = {'data_matrix': lgs_matrix, 'edges': lgs_edge_index}
    
    #lncRNA cosine similarity
    lcs_matrix = read_csv(args.dataset_path + '/LCS.csv')
    lcs_edge_index = get_edge_index(lcs_matrix)
    dataset['lcs'] = {'data_matrix': lcs_matrix, 'edges': lcs_edge_index}

    return dataset

