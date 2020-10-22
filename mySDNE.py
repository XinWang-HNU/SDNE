import sys
path_result = "./Latent_Representation/"
from Model_SDNE import *
from Metrics import *
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time

import warnings
warnings.filterwarnings('ignore')


#  features: X (n × d);
#  adjacency matrix:N *N;
#  labels: Y

######################################################### Setting #####################################################
Dataset = 'cora'
Classification = True
Clustering = False
t_SNE = True
########################################## hyper-parameters##############################################################
Epoch_Num = 200
Learning_Rate = 5*1e-4

Hidden_Layer_1 = 1024
Hidden_Layer_2 = 128
################################### Load dataset   ######################################################################
if (Dataset is "cora") or (Dataset is "citeseer"):
    load_data = Load_Data(Dataset)
    Features, Labels, Adjacency_Matrix = load_data.Graph()
    Features = torch.Tensor(Features)
    Adjacency_Matrix = torch.Tensor(Adjacency_Matrix)
else:
    load_data = Load_Data(Dataset)
    Features, Labels = load_data.CPU()
    Features = torch.Tensor(Features)
################################### Calculate the adjacency matrix #########################################################
if('Adjacency_Matrix' in vars()):
    print('Adjacency matrix is raw')
    pass
else:
    print('Adjacency matrix is caculated by KNN')
    graph = Graph_Construction(Features)
    Adjacency_Matrix = graph.KNN()
################################################ adjacency convolution ##################################################
convolution_kernel = Convolution_Kernel(Adjacency_Matrix)
Laplacian_Convolution = convolution_kernel.Laplacian_Convolution()
########################################## hyper-parameters##############################################################
Epoch_Num = 40
Learning_Rate = 1e-4
Lambda = 1

Input_Dim = Adjacency_Matrix.shape[0]
Hidden_Layer_1 = 1024
Hidden_Layer_2 = 128

B = Adjacency_Matrix * (20 - 1) + 1
############################################ Results  Initialization ###################################################
ACC_SDNE_total = []
NMI_SDNE_total = []
PUR_SDNE_total = []

ACC_SDNE_total_STD = []
NMI_SDNE_total_STD = []
PUR_SDNE_total_STD = []

F1_score = []

#######################################  Model #########################################################################
mse_loss = torch.nn.MSELoss(size_average=False)
model_SDNE = mySDNE(Input_Dim, Hidden_Layer_1, Hidden_Layer_2)
optimzer = torch.optim.Adam(model_SDNE.parameters(), lr=Learning_Rate)
#######################################  Train and result ################################################################
for epoch in range(Epoch_Num):
    Latent_Representation, Graph_Reconstrction = model_SDNE(Adjacency_Matrix)
    loss_1st = torch.norm((Graph_Reconstrction - Adjacency_Matrix)*B, p = 'fro')
    loss_2st = torch.trace((Latent_Representation.T).mm(Laplacian_Convolution).mm(Latent_Representation))
    loss = loss_1st + Lambda * loss_2st

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Latent_Representation.cpu().detach().numpy()
    ##################################################### Results  ####################################################
    if Classification and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = mySVM(Latent_Representation, Labels, scale=0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, Epoch_Num, score))
        F1_score.append(score)
        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)

    elif Clustering and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))

        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        ##############################################################################
        kmeans = KMeans(n_clusters=max(np.int_(Labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(Latent_Representation)
            Y_pred_OK = np.array(Y_pred_OK)
            Labels = np.array(Labels).flatten()
            AM = clustering_metrics(Y_pred_OK, Labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        ACC_SDNE_total.append(100 * np.mean(ACC_H2))
        NMI_SDNE_total.append(100 * np.mean(NMI_H2))
        PUR_SDNE_total.append(100 * np.mean(PUR_H2))

        ACC_SDNE_total_STD.append(100 * np.std(ACC_H2))
        NMI_SDNE_total_STD.append(100 * np.std(NMI_H2))
        PUR_SDNE_total_STD.append(100 * np.std(PUR_H2))
        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
##################################################  Result #############################################################
if Clustering:

    Index_MAX = np.argmax(ACC_SDNE_total)

    ACC_SDNE_max = np.float(ACC_SDNE_total[Index_MAX])
    NMI_SDNE_max = np.float(NMI_SDNE_total[Index_MAX])
    PUR_SDNE_max = np.float(PUR_SDNE_total[Index_MAX])

    ACC_STD = np.float(ACC_SDNE_total_STD[Index_MAX])
    NMI_STD = np.float(NMI_SDNE_total_STD[Index_MAX])
    PUR_STD = np.float(PUR_SDNE_total_STD[Index_MAX])

    print('ACC_SDNE_max={:.2f} +- {:.2f}'.format(ACC_SDNE_max, ACC_STD))
    print('NMI_SDNE_max={:.2f} +- {:.2f}'.format(NMI_SDNE_max, NMI_STD))
    print('PUR_SDNE_max={:.2f} +- {:.2f}'.format(PUR_SDNE_max, PUR_STD))

elif Classification:
    Index_MAX = np.argmax(F1_score)
    print("SDNE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

########################################################### t- SNE #################################################
if t_SNE:
    print("dataset is {}".format(Dataset))
    Latent_Representation_max = np.load(path_result + "{}.npy".format((Index_MAX+1) * 5))
    Features = np.array(Features)
    plot_embeddings(Latent_Representation_max, Features, Labels)