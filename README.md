This is the implementation of paper 'Structural Deep Network Embedding' in ACM SIGKDD, 2016.

1. This is the SDNE I reproduced using PyTorch.

2. There are three tasks used to evaluate the effect of network embedding, i.e., node clustering, node classification, and graph Visualization.

3. Algorithms used in the tasks:

      Clustering：k-means; 
      Classification: SVM; 
      Visualization: t-SNE;

4. Requirement: Python 3.7, Pytorch: 1.5 and other pakeages which is illustrated in the code. And the codes can be runned in the windows.

5. The purpose of the SDNE is mainly to learn the latent representation, etc, the network embedding. There are two datasets in this example，i.e., Cora and Yale. If you want to use other datasets, you just need to put your dataset in the "Dataset" folder. Cora is a graph dataset and ATT is a no-graph dataset. The adjacency matrix for the ATT is calculated by the KNN (k=9).

6. If you think my code is helpful to you, please light me a star, thank you.
