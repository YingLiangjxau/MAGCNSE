import torch
from torch import nn
from torch_geometric.nn import GCNConv
torch.backends.cudnn.enabled = False
import csv
import numpy
import pandas as pd


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class MAGCN(nn.Module):
    def __init__(self, args):
        super(MAGCN, self).__init__()
        self.args = args

        self.gcn_x1_lfs = GCNConv(self.args.fl, self.args.fl)  #  define the first GCN layer to tackle lncRNA_view 1
        self.gcn_x1_lgs = GCNConv(self.args.fl, self.args.fl)  #  define the first GCN layer to tackle lncRNA_view 2
        self.gcn_x1_lcs = GCNConv(self.args.fl, self.args.fl)  #  define the first GCN layer to tackle lncRNA_view 3
        
        self.gcn_x2_lfs = GCNConv(self.args.fl, self.args.fl)  #  define the second GCN layer to tackle lncRNA_view 1
        self.gcn_x2_lgs = GCNConv(self.args.fl, self.args.fl)  #  define the second GCN layer to tackle lncRNA_view 2
        self.gcn_x2_lcs = GCNConv(self.args.fl, self.args.fl)  #  define the second GCN layer to tackle lncRNA_view 3
         
        self.gcn_y1_dss = GCNConv(self.args.fd, self.args.fd)  #  define the first GCN layer to tackle disease_view 1
        self.gcn_y1_dgs = GCNConv(self.args.fd, self.args.fd)  #  define the first GCN layer to tackle disease_view 2
        self.gcn_y1_dcs = GCNConv(self.args.fd, self.args.fd)  #  define the first GCN layer to tackle disease_view 3
        
        
        self.gcn_y2_dss = GCNConv(self.args.fd, self.args.fd)  #  define the second GCN layer to tackle disease_view 1
        self.gcn_y2_dgs = GCNConv(self.args.fd, self.args.fd)  #  define the second GCN layer to tackle disease_view 2
        self.gcn_y2_dcs = GCNConv(self.args.fd, self.args.fd)  #  define the second GCN layer to tackle disease_view 3


        # define the network structure in attention mechanism module
        self.globalAvgPool_x = nn.AvgPool2d((self.args.fl, self.args.lncRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))        

        self.fc1_x = nn.Linear(in_features=self.args.lncRNA_view*self.args.gcn_layers,
                             out_features=5*self.args.lncRNA_view*self.args.gcn_layers) 
        self.fc2_x = nn.Linear(in_features=5*self.args.lncRNA_view*self.args.gcn_layers,
                             out_features=self.args.lncRNA_view*self.args.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.args.disease_view * self.args.gcn_layers,
                             out_features=5 * self.args.disease_view * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.disease_view * self.args.gcn_layers,
                             out_features=self.args.disease_view * self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()
        self.temp=0;

        # define the network structure in convolutional neural network module
        self.cnn_x = nn.Conv1d(in_channels=self.args.lncRNA_view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fl, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.disease_view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        x_l = torch.randn(self.args.lncRNA_number, self.args.fl)
        x_d = torch.randn(self.args.disease_number, self.args.fd)

        x_l_lfs1 = torch.relu(self.gcn_x1_lfs(x_l.cuda(), data['lfs']['edges'].cuda(), data['lfs']['data_matrix'][data['lfs']['edges'][0], data['lfs']['edges'][1]].cuda()))
        x_l_lfs2 = torch.relu(self.gcn_x2_lfs(x_l_lfs1, data['lfs']['edges'].cuda(), data['lfs']['data_matrix'][data['lfs']['edges'][0], data['lfs']['edges'][1]].cuda()))

        x_l_lgs1 = torch.relu(self.gcn_x1_lgs(x_l.cuda(), data['lgs']['edges'].cuda(), data['lgs']['data_matrix'][data['lgs']['edges'][0], data['lgs']['edges'][1]].cuda()))
        x_l_lgs2 = torch.relu(self.gcn_x2_lgs(x_l_lgs1, data['lgs']['edges'].cuda(), data['lgs']['data_matrix'][data['lgs']['edges'][0], data['lgs']['edges'][1]].cuda()))

        x_l_lcs1 = torch.relu(self.gcn_x1_lcs(x_l.cuda(), data['lcs']['edges'].cuda(), data['lcs']['data_matrix'][data['lcs']['edges'][0], data['lcs']['edges'][1]].cuda()))
        x_l_lcs2 = torch.relu(self.gcn_x2_lcs(x_l_lcs1, data['lcs']['edges'].cuda(), data['lcs']['data_matrix'][data['lcs']['edges'][0], data['lcs']['edges'][1]].cuda()))

        y_d_dss1 = torch.relu(self.gcn_y1_dss(x_d.cuda(), data['dss']['edges'].cuda(), data['dss']['data_matrix'][data['dss']['edges'][0], data['dss']['edges'][1]].cuda()))
        y_d_dss2 = torch.relu(self.gcn_y2_dss(y_d_dss1, data['dss']['edges'].cuda(), data['dss']['data_matrix'][data['dss']['edges'][0], data['dss']['edges'][1]].cuda()))

        y_d_dgs1 = torch.relu(self.gcn_y1_dgs(x_d.cuda(), data['dgs']['edges'].cuda(), data['dgs']['data_matrix'][data['dgs']['edges'][0], data['dgs']['edges'][1]].cuda()))
        y_d_dgs2 = torch.relu(self.gcn_y2_dgs(y_d_dgs1, data['dgs']['edges'].cuda(), data['dgs']['data_matrix'][data['dgs']['edges'][0], data['dgs']['edges'][1]].cuda()))

        y_d_dcs1 = torch.relu(self.gcn_y1_dcs(x_d.cuda(), data['dcs']['edges'].cuda(), data['dcs']['data_matrix'][data['dcs']['edges'][0], data['dcs']['edges'][1]].cuda()))
        y_d_dcs2 = torch.relu(self.gcn_y2_dcs(y_d_dcs1, data['dcs']['edges'].cuda(), data['dcs']['data_matrix'][data['dcs']['edges'][0], data['dcs']['edges'][1]].cuda()))



        XM = torch.cat((x_l_lfs1, x_l_lfs2, x_l_lgs1, x_l_lgs2, x_l_lcs1, x_l_lcs2), 1).t()
        XM = XM.view(1, self.args.lncRNA_view*self.args.gcn_layers, self.args.fl, -1)
        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        YD = torch.cat((y_d_dss1, y_d_dss2, y_d_dgs1, y_d_dgs2, y_d_dcs1, y_d_dcs2), 1).t()
        YD = YD.view(1, self.args.disease_view*self.args.gcn_layers, self.args.fd, -1)
        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD
        YD_channel_attention = torch.relu(YD_channel_attention)

        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.lncRNA_number).t()
        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()

        self.temp=self.temp+1
        # obtain the final representation of lncRNAs and diseases
        if self.temp==self.args.epoch:
            savex=x.cpu().detach().numpy()
            save_x=pd.DataFrame(savex)
            save_x.to_csv('lncRNAFeature.csv')
            savey=y.cpu().detach().numpy()
            save_y=pd.DataFrame(savey)
            save_y.to_csv('diseaseFeature.csv')
        return x.mm(y.t())


