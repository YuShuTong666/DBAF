import dgl
import numpy as np
from dgl.data import DGLDataset
import torch
import os
import random
import torchvision.transforms as T
from torchvision.io import read_image
from dgl.nn.functional import edge_softmax
import dgl.function as fn

class ImageGraph(DGLDataset):
    '''
    Build a graph represents an image with shape (3, n, n)
    '''
    def __init__(self, n, T=0.00002, delta=0.5):
        self.n = n
        self.T = T
        self.delta = delta
        super().__init__(name='image_graph')

    def process(self):
        n = self.n
        edges_src = torch.tensor([], dtype=torch.int32)
        edges_dst = torch.tensor([], dtype=torch.int32)
        # Build the edges between left pixels and right pixels of the first channel
        src = torch.tensor([j for j in range(0, n * n) if (j + 1) % n != 0])
        edges_src = torch.cat((edges_src, src))
        dst = src + 1
        edges_dst = torch.cat((edges_dst, dst))
        # Build the edges between upper pixels and lower pixels of the first channel
        src = torch.tensor([j for j in range(0, n * (n - 1))])
        edges_src = torch.cat((edges_src, src))
        dst = src + n
        edges_dst = torch.cat((edges_dst, dst))

        # Build the edges between upper left pixels and lower right pixels of the first channel
        src = torch.tensor([j for j in range(0, n * (n - 1)) if (j + 1) % n != 0])
        edges_src = torch.cat((edges_src, src))
        dst = src + n + 1
        edges_dst = torch.cat((edges_dst, dst))

        # Build the edges between lower left pixels and lower right pixels of the first channel
        src = torch.tensor([j for j in range(0, n * (n - 1)) if j % n != 0])
        edges_src = torch.cat((edges_src, src))
        dst = src + n - 1
        edges_dst = torch.cat((edges_dst, dst))

        # Build the edges between pixels in the 2th and 3th channel
        edges_src = torch.cat((edges_src, edges_src + n * n, edges_src + n * n * 2))
        edges_dst = torch.cat((edges_dst, edges_dst + n * n, edges_dst + n * n * 2))
        '''# Build the edges between channels
        src = torch.tensor([j for j in range(0, n * n)])
        edges_src = torch.cat((edges_src, src, src, src + n * n))
        edges_dst = torch.cat((edges_dst, src + n * n, src + 2 * n * n, src + 2 * n * n))'''
        # Build the graph
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=3 * n * n)
        self.graph = dgl.to_bidirected(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("graph on gpu")
        else:
            device = torch.device("cpu")
            print("graph on cpu")
        #self.graph.ndata['train_mask'] = torch.zeros(n*n*3, dtype=torch.bool)
        #self.graph.ndata['label'] = torch.zeros(n*n*3)
        self.graph = self.graph.to(device)

        '''self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask'''

    def reprocess(self, features=None, k=0):

        '''n = self.n
        arr = list(range(n*n*3))
        random.shuffle(arr)
        mask = torch.zeros(n*n*3, dtype=torch.bool)
        mask[arr[:k]] = True
        if torch.cuda.is_available():
            mask = mask.cuda()
        #train_mask = torch.tensor(arr[:k], dtype=torch.int32)
        self.graph.ndata['train_mask'] = mask'''
        #print(features.shape)
        assert features.shape[0] == 3 and features.shape[1] == self.n and features.shape[2] == self.n
        features = features.reshape(-1, 1)
        #self.graph.ndata['feat'] = features
        #print(self.graph.edges()[0])
        #edges = [torch.abs(features[self.graph.edges()[0][i]] - features[self.graph.edges()[1][i]]) for i in range(self.graph.edges()[0].shape[0])]
        #edges = torch.cos(torch.square(features[self.graph.edges()[0]] - features[self.graph.edges()[1]])).float()
        #edges = torch.cos(features[self.graph.edges()[0]] - features[self.graph.edges()[1]]).float()
        edges = (1.0-torch.square(features[self.graph.edges()[0]] - features[self.graph.edges()[1]])).float()
        edges = edges.cuda()
        edges = edges / self.T
        edges_softmax = edge_softmax(self.graph, edges)
        self.graph.edata['h'] = edges_softmax
        #print(edges_softmax[:10])
        '''self.mask = None
        with self.graph.local_scope():
            self.graph.update_all(fn.copy_e('h', 'm'), fn.max('m', 'h_max'))
            self.mask = torch.where(self.graph.ndata['h_max'] > self.delta, 1, 0)'''
        #print(self.mask)

    def set_labels(self):
        label = torch.from_numpy(np.random.normal(0., 1., (self.n*self.n*3, 1))).to(torch.float32)
        if torch.cuda.is_available():
            label = label.cuda()
        self.graph.ndata['label'] = label

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    path = '../data/imagenet_07_609.jpg'
    image = read_image(path).cuda()
    image = image / 255
    dataset = ImageGraph(512)
    graph = dataset[0]

    dataset.reprocess(image)