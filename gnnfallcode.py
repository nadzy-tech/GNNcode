#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn

import dgl
import dgl.data
import networkx as nx


# In[67]:


#dataset = dgl.data.CoraGraphDataset()
import pandas as pd
import dgl

# Load the data from a CSV file
data = pd.read_csv('/Users/nadiaahmed/Documents/3RDYEAR/DISSERTATION/Subject1Activity1Trial1COMB.csv')

# Convert the DataFrame to a DGL graph
# src = data['AnkleAccelerometer'].tolist()
# dst = data['Subject'].tolist()
# g = dgl.graph((src, dst))
print(data)
# print(src)
# print(g.ndata)

# Create a DGL graph
src = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])
dst = torch.tensor([2, 1, 3, 4, 0, 0, 0, 0])
g = dgl.graph((src, dst))
print(g)
print(g.srcnodes())
print(g.dstnodes())

node_feat = torch.stack((torch.tensor(data['AnkleAngularVelocity'].values), 
                     torch.tensor(data['RightPocketAngularVelocity'].values),
                    torch.tensor(data['BeltAngularVelocity'].values),
                   torch.tensor(data['NeckAngularVelocity'].values),
                    torch.tensor(data['WristAngularVelocity'].values)
                   ), dim=0)
print('node_feat', node_feat.shape)
g.ndata['velocity'] = node_feat
print('2', g.ndata['velocity'])


    # Set the edge features
test = torch.stack((torch.tensor(data['AnkleAccelerometer'].values), 
                     torch.tensor(data['RightPocketAccelerometer'].values),
                   torch.tensor(data['NeckAccelerometer'].values),
                    torch.tensor(data['WristAccelerometer'].values),
                    torch.tensor(data['AnkleAccelerometer'].values), 
                     torch.tensor(data['RightPocketAccelerometer'].values),
                   torch.tensor(data['NeckAccelerometer'].values),
                    torch.tensor(data['WristAccelerometer'].values)
                    ), dim=0)
print('test', test.shape)
g.edata['Accelerometer'] = test
print('1', g.edata['Accelerometer'])
#g.edata['AnkleAccelerometer'] = torch.FloatTensor(data['AnkleAccelerometer'].values)
# g.edata['RightPocketAccelerometer'] = torch.FloatTensor(data['RightPocketAngularVelocity'].values)
# g.edata['BeltAccelerometer'] = torch.FloatTensor(data['BeltAngularVelocity'].values)
# g.edata['NeckAccelerometer'] = torch.FloatTensor(data['NeckAngularVelocity'].values)
# g.edata['WristAccelerometer'] = torch.FloatTensor(data['WristAngularVelocity'].values)


# In[68]:


print("Node features")
print(g.ndata)
print("Edge features")
print(g.edata)


# In[69]:


from dgl.nn import GraphConv
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# Create the model with given dimensions

model = GCN(g.ndata["velocity"].shape[1], 16, 3)

print(g.srcnodes())
print(g.dstnodes())

#output = read edge and node features and convert them to a pytorch tensor


# In[70]:
###Graph Visualisation

# def visualize_graph(g, color):
#     plt.figure(figsize=(12,12))
#     plt.xticks([])
#     plt.yticks([])
#     nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
#                      node_color=color, cmap="Set2")
#     plt.show()
# import matplotlib.pyplot as plt
# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }
# G = dgl.to_networkx(g)
# plt.figure(figsize=[15,7])
# nx.draw(G, **options)


import numpy as np
from sklearn.model_selection import train_test_split
# train_mask = data.Activity
# val_mask = data.Activity
# test_mask = data.Activity

sequence = range(len(data.Activity.values))

def spliting():
    train_set, test_set = train_test_split(data.values, test_size=0.3)
    return train_set, test_set

#calling the function
train_set, test_set = spliting()

data = pd.read_csv('/Users/nadiaahmed/Documents/3RDYEAR/DISSERTATION/Subject1Activity1Trial1COMB.csv', skiprows=1)

def graph_generation(train_set, test_set):
#     a = train_set[:, 4:7]
#     node_feat = torch.stack((torch.tensor(train_set[:,4:7].astype(np.float32)), #AnkleAngularVelocity
#     torch.tensor(train_set[:,11:14].astype(np.float32)), #RightPocketAngularVelocity
#     torch.tensor(train_set[:,18:21].astype(np.float32)), #BeltAngularVelocity
#     torch.tensor(train_set[:,25:28].astype(np.float32)), #NeckAngularVelocity
#     torch.tensor(train_set[:,32:35].astype(np.float32))), dim=0) #WristAngularVelocity)

    node_feat = torch.stack((torch.tensor(train_set[:,1:8].astype(np.float32)), #ALLAnkle
    torch.tensor(train_set[:,8:15].astype(np.float32)), #RightPocketAngularVelocity
    torch.tensor(train_set[:,15:22].astype(np.float32)), #BeltAngularVelocity
    torch.tensor(train_set[:,22:29].astype(np.float32)), #NeckAngularVelocity
    torch.tensor(train_set[:,29:36].astype(np.float32))), dim=0) #WristAngularVelocity)
    node_feat = torch.transpose(node_feat, 0, 1)

#     train_edge_feat = torch.stack((torch.tensor(train_set[:,1:4].astype(np.float32)), 
#     torch.tensor(train_set[:,8:11].astype(np.float32)), #RightPocketAccelerometer
#     torch.tensor(train_set[:,22:25].astype(np.float32)), #NeckAccelerometer
#     torch.tensor(train_set[:,15:18].astype(np.float32)), #Belt
#     torch.tensor(train_set[:,1:4].astype(np.float32)), #AnkleAccelerometer
#     torch.tensor(train_set[:,9:12].astype(np.float32)), #rightpocket
#     torch.tensor(train_set[:,20:23].astype(np.float32)), #NeckAccelerometer
#     torch.tensor(train_set[:,29:32].astype(np.float32))),dim=0) #WristAccelerometer)
    
#     test_node_feat = torch.stack((torch.tensor(test_set[:,4:7].astype(np.float32)), 
#     torch.tensor(test_set[:,11:14].astype(np.float32)),
#     torch.tensor(test_set[:,18:21].astype(np.float32)),
#     torch.tensor(test_set[:,25:28].astype(np.float32)),
#     torch.tensor(test_set[:,32:35].astype(np.float32))),dim=0)

    test_node_feat = torch.stack((torch.tensor(test_set[:,1:8].astype(np.float32)), #ALLAnkle
    torch.tensor(test_set[:,8:15].astype(np.float32)), #RightPocketAngularVelocity
    torch.tensor(test_set[:,15:22].astype(np.float32)), #BeltAngularVelocity
    torch.tensor(test_set[:,22:29].astype(np.float32)), #NeckAngularVelocity
    torch.tensor(test_set[:,29:36].astype(np.float32))), dim=0) #WristAngularVelocity)
    test_node_feat = torch.transpose(test_node_feat, 0, 1)
    
#     test_edge_feat = torch.stack((torch.tensor(test_set[:,1:4].astype(np.float32)), 
#     torch.tensor(test_set[:,8:11].astype(np.float32)),
#     torch.tensor(test_set[:,22:25].astype(np.float32)),
#     torch.tensor(test_set[:,15:18].astype(np.float32)),
#     torch.tensor(test_set[:,1:4].astype(np.float32)), 
#     torch.tensor(test_set[:,9:12].astype(np.float32)),
#     torch.tensor(test_set[:,20:23].astype(np.float32)),
#     torch.tensor(test_set[:,29:32].astype(np.float32))), dim=0)

    train_mask = train_set[:, 44].astype(np.float32)
    test_mask = test_set[:, 44].astype(np.float32)

    # return node_feat, train_edge_feat, test_node_feat, test_edge_feat
    return node_feat, test_node_feat, train_mask, test_mask
# #calling
#  node_feat, train_edge_feat, test_node_feat, test_edge_feat = graph_generation(train_set, test_set)
# node_feat, test_node_feat = graph_generation(train_set, test_set)
    
# model = GCN(g.ndata["velocity"].shape[1], 16, g.num_edges())
# train(g, model)

######TRAINING THE MODEL
#FIX THESE BUGS, DEBUG HERE AND ACHIEVE PREDICTIONS
#generate graph, model, predictions

import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(torch.nn.Module):
    def __init__(self, num_feat, num_hidden, num_hidden1, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(num_feat, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden1)
        self.layer = nn.Linear(5 * num_hidden1, num_classes)

    def forward(self, g, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float)
        h = torch.relu(self.conv1(g, inputs))
        h = torch.relu(self.conv2(g, h))
        h = h.flatten()
        h = self.layer(h)
        # h = torch.softmax(h, dim=0)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05) #The function first initializes an Adam optimizer 
    #with a learning rate of 0.01 and sets two variables to keep track of the best validation and test accuracies seen 
    #during training

#making lists here
    best_val_acc = 0
    best_test_acc = 0
    loss_list = []
    epoch_number = []
    accuracy_list = []
    train_acc_list = []
    test_acc_list = []

    # dataset is a node classification dataset, need masks indicating whether a node belongs to training, val and test set.
    # val_mask and test_mask, which denotes which nodes should be used for validation and testing.
    # nodes = g.ndata['velocity']
    # labels = g.edata['Accelerometer']
    
# train_mask
#     train_mask = g.ndata['train_mask'] #sample_activity
#     val_mask = g.ndata["val_mask"]
#     test_mask = g.ndata["test_mask"]

    for e in range(150):
        # Forward
        train_set, test_set = spliting()
        node_feat, test_node_feat, train_mask, test_mask = graph_generation(train_set, test_set)
        logits = []
        pred = []
        #accumalate loss into one big loss
        for i in range(len(node_feat)):
            # features = g.ndata["velocity"]
            logits.append(model(g, node_feat[i]))

            # Compute prediction
            # pred.append(logits.argmax(0) + 1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
               
        logits = torch.stack((logits), dim=0)
        #train_mask = torch.nn.functional.one_hot(torch.tensor(train_mask, dtype=torch.float64), num_classes=3)
        #train_mask = torch.nn.functional.one_hot(train_mask.clone().detach()(train_mask, dtype=torch.float64), num_classes=3)
        #inputs = train_mask.clone().detach()
        train_mask = torch.tensor(train_mask, dtype=torch.long) - 1
        # train_mask_tensor = train_mask_tensor.clone().detach()
        train_mask_one_hot = torch.nn.functional.one_hot(train_mask, num_classes=3)


        pred = logits.argmax(1)
        loss = F.cross_entropy(logits, train_mask_one_hot.float())
        loss_list.append(loss.detach().numpy())
        epoch_number.append(e)
        
        
        #Compute accuracy on training/validation/test
        train_acc = (pred == train_mask).float().mean()
        

        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_logits = []
        for i in range(len(test_node_feat)):
            # features = g.ndata["velocity"]
            test_logits.append(model(g, test_node_feat[i]))

            # Compute prediction
            # pred.append(logits.argmax(0) + 1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
               
        test_logits = torch.stack((test_logits), dim=0)

        test_pred = test_logits.argmax(1)
        test_mask = torch.tensor(test_mask, dtype=torch.long) - 1
        test_acc = (test_pred == test_mask).float().mean()

        train_acc_list.append(train_acc.detach().numpy())
        test_acc_list.append(test_acc.detach().numpy())

        print(loss, train_acc, test_acc)


    return epoch_number, loss_list, train_acc_list, test_acc_list

model = GCN(7, 16, 8, 3)
epoch_number, loss_list, train_acc_list, test_acc_list = train(g, model)
import matplotlib.pyplot as plt

def plot_data(x, y, title='', x_label='', y_label=''):
    """Plot x_data against y_data using matplotlib."""
    
    # create a new figure
    plt.figure()
    
    # plot the data
    plt.plot(x, y)
    
    # add a title and axis labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # show the plot
    plt.show()

plot_data(epoch_number, loss_list, title='Training Loss', x_label='epoch', y_label='Loss')

plot_data(epoch_number, train_acc_list, title='Training Accuracy', x_label='epoch', y_label='Training Accuracy')
plot_data(epoch_number, test_acc_list, title='Test Accuracy', x_label='epoch', y_label='Testing Accuracy')

#expect one hot (OH) matrix as output; follow youtube video

# In[21]:

#Calculate Accuracy
from sklearn.metrics import accuracy_score
# accuracy(y_test, y_pred)
# Accuracy is 0.88124999999999998

# #Recall is 0.84109589041095889
# Calculate Precision
# from sklearn.metrics import precision_score
# precision_score(y_test, y_pred)
# Precision is 0.84573002754820936


# In[53]:
#g = g.to('cuda')
#model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
#train(g, model)
# x_train = data.to_numpy()
# history = data(g, x_train,y)


# #Plotting the learning curves.

# display_learning_curves(history)


















