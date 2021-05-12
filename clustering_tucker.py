#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

import os
from PIL import Image

from Functions import *
from tqdm import tqdm

import adios2 as ad2
import xgc4py
import nanopq
import sys


# In[2]:


def draw(frame):
    x = np.linspace(0, 38, 39)
    y = np.linspace(0, 38, 39)
    X, Y = np.meshgrid(x, y)
    plt.imshow(frame, origin='lower')
    #plt.imshow(frame, origin='upper')
    plt.colorbar()
    plt.contour(X, Y, frame, 5, origin='image', colors='white', alpha=0.5)


# In[3]:


with ad2.open('./d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz = f.read('rz')
    conn = f.read('nd_connect_list')
    psi = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

r = rz[:,0]
z = rz[:,1]
print (nnodes)


# In[4]:


with ad2.open('./d3d_coarse_v2/restart_dir/xgc.f0.00420.bp', 'r') as f:
    i_f = f.read('i_f')


# In[5]:


data_load = np.moveaxis(i_f, 1, 2)
print(data_load.shape)


# In[6]:


xgcexp = xgc4py.XGC('./d3d_coarse_v2')


# In[7]:


#import f0_diag_test
#xgc = f0_diag_test.XGC_f0_diag(xgcexp)


# In[8]:


all_planes = np.copy(data_load)

# changed for separate tucker

mu = np.mean(all_planes, axis=(2,3))
sig = np.std(all_planes, axis=(2,3))
all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis]
print(all_planes.shape, mu.shape, sig.shape)


# In[38]:


# In[46]:


labels_tania2 = np.loadtxt(sys.argv[1])
labels_tania2
#labels_tania2[labels_tania2>0] = 1
#labels_tania2[labels_tania2==0] = 2
#labels_tania2[labels_tania2==1] = 0


# In[110]:


print (np.count_nonzero(labels_tania2 == 0), np.count_nonzero(labels_tania2 == 1))


# In[111]:

labels = labels_tania2.copy()
'''
labels = np.zeros((16395), dtype = int)
print(labels.shape)
for i in range(16395):
    if (labels_tania2[i] == 0):
        labels[i] = 0
    #elif ((labels_tania[i] != 0) and (labels_tania[i] != 775)):
    elif ((labels_tania2[i] != 0) and (labels_tania2[i] != 59)):
        labels[i] = 1
    elif (labels_tania2[i] == 59):
        labels[i] = 2
''' 


# In[112]:


print (np.count_nonzero(labels == 0), np.count_nonzero(labels == 1))


# In[113]:


n_comp = 1 #len(np.unique(labels))

cluster = []
index_cluster = []

temp = np.moveaxis(all_planes, 0,1)
print(temp.shape)
'''
for i in range(n_comp):
    clus = []
    idx = []
    for j in range(temp.shape[0]):
        if (labels[j] == i):
            temp1 = temp[j]
            clus.append(temp1)
            idx.append(j)
    clus = np.array(clus)
    clus = np.moveaxis(clus, 0,1)
    print(clus.shape)
    cluster.append(clus)
    
    idx = np.array(idx)
    index_cluster.append(idx)
''' 
clus = []
idx = []
for j in range(temp.shape[0]):
    clus.append(temp[j])
    idx.append(j)

clus = np.array(clus)
clus = np.moveaxis(clus, 0,1)
print(clus.shape)
cluster.append(clus)
idx = np.array(idx)
index_cluster.append(idx)

#del temp, temp1


# In[114]:


#draw(cluster[1][0][1500])


# In[115]:


#draw(all_planes[0][index_cluster[1][1500]])


# In[116]:


Epochs = 100
Lr_Rate = 1e-3
Batch_Size = 128


# In[117]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent):
        super(Autoencoder, self).__init__()
        
        self.enc1 = nn.Linear(in_features=input_dim, out_features=latent)
    
    def encode(self, x):
        x = self.enc1(x)        
        return x
    
    def decode(self, x):        
        x = F.linear(x, weight=self.enc1.weight.transpose(0,1))        
        return x

    def forward(self, x):
        
        latent = self.encode(x)
        recon = self.decode(latent)

        return recon


# In[118]:


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# In[119]:


def training(model, train_loader, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            data = data[0].to(device)            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))

    return train_loss


# In[120]:


def test_reconstruct(model, test_loader, num_instance, input_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    reconstructions = []
    
    temp = 0
    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model(data)
        if (temp == 0):
            print('Tensor type:', outputs[0][0].dtype)
            temp += 1
        reconstructions.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
        
    decoded_vectors = np.zeros((num_instance,input_dim), dtype=np.float32)
    count = 0
    for i in range(len(reconstructions)):
        for j in range(len(reconstructions[i])):
            decoded_vectors[count] = np.float32(reconstructions[i][j])
            count = count + 1
    
    return decoded_vectors


# In[121]:


def encoding(model, test_loader, num_instance, latent_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    latent_vectors = []
    
    #count = 0
    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model.encode(data)
        latent_vectors.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
    
    encoded_vectors = np.zeros((num_instance,latent_dim), dtype=np.float32)
    count = 0
    for i in range(len(latent_vectors)):
        for j in range(len(latent_vectors[i])):
            encoded_vectors[count] = np.float32(latent_vectors[i][j])
            count = count + 1  
    
    return encoded_vectors


# In[122]:


def decoding(model, test_loader, num_instance, input_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    reconstructions = []    

    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model.decode(data)
        reconstructions.append(outputs.cpu().data.numpy())
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
    
    decoded_vectors = np.zeros((num_instance,input_dim), dtype=np.float32)
    count = 0
    for i in range(len(reconstructions)):
        for j in range(len(reconstructions[i])):
            decoded_vectors[count] = np.float32(reconstructions[i][j])
            count = count + 1
    
    return decoded_vectors


# In[123]:


def rmse_error(x, y):

    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)

    return np.sqrt(mse)


# In[124]:


# Tucker AE - Per Cluster and per plane, Full Tucker coeffs, Tania Clustering

latent_dims = [32, 64, 128, 256]

TuckerAE_cluster_all = []

for c in range(n_comp):
    TuckerAE_cluster = []
    
    for p in range(1):
    
        TuckerAE_cluster_plane = []
    
        X_train = cluster[c][p]

        tucker_data = np.array(X_train).reshape([X_train.shape[0],39,39])
        print(tucker_data.shape)
    
        tqdm.write('converting data to tucker basis...')
        coeff_tucker, factors = tucker(tucker_data)

        experiments = [
            {
                "num_coeffs": 1521
            }
        ]    

        for experiment in experiments:
            num_coeffs = experiment['num_coeffs']
        
            tqdm.write('tucker_coeffs: '+str(num_coeffs))

            top_coeffs = tucker_top_coeffs(coeff_tucker, num_coeffs)

            X_train_tucker = to_tucker_basis(np.array(X_train).reshape([X_train.shape[0],39,39]), factors, top_coeffs)
        
            print(X_train.shape)
            print(X_train_tucker.shape)
        
            training_data_tucker = torch.utils.data.TensorDataset(torch.Tensor(X_train_tucker))
        
            training_loader_tucker = DataLoader(training_data_tucker,
                                                batch_size=Batch_Size,
                                                shuffle=False,
                                                pin_memory=True)
        
        
            for j in range(len(latent_dims)):
                latent_dim = latent_dims[j]
                model = Autoencoder(X_train_tucker.shape[1], latent_dim)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
                device = get_device()
                model.to(device)
                print(model)
            
                train_loss = training(model, training_loader_tucker, Epochs)
        
                latent_train = encoding(model, training_loader_tucker, X_train.shape[0], latent_dim) 
                print('latent train shape: ', latent_train.shape)
        
        
                def train_quantizer():
                    seed = np.random.randint(100)
                    pq = nanopq.PQ(M=latent_dim, Ks=256, verbose=True)
                    pq.fit(vecs=latent_train, iter=20, seed=seed)
                    return pq

                pq = train_quantizer()
                latent_train_quan = pq.encode(vecs=latent_train)
        
                latent_train_dequan = pq.decode(codes=latent_train_quan)
            
                training_latent = torch.utils.data.TensorDataset(torch.Tensor(latent_train_dequan))
        
                training_loader_latent = DataLoader(training_latent,
                                                    batch_size=Batch_Size,
                                                    shuffle=False,
                                                    pin_memory=True)
        
                recon_train_tucker = decoding(model, training_loader_latent, X_train.shape[0], X_train_tucker.shape[1])
        
                print('recon train shape: ', recon_train_tucker.shape)
            
        
                train_recon = from_tucker_basis(recon_train_tucker, factors, top_coeffs, data_shape=[X_train.shape[0],39,39])
                #train_orig_images = np.array(X_train).reshape([X_train.shape[0], 39,39])
                TuckerAE_cluster_plane.append(train_recon)
            TuckerAE_cluster.append(TuckerAE_cluster_plane)

    TuckerAE_cluster_all.append(TuckerAE_cluster)


# In[128]:


print(len(TuckerAE_cluster_all), len(TuckerAE_cluster_all[0]), len(TuckerAE_cluster_all[0][0]), len(TuckerAE_cluster_all[0][0][0]))


# In[129]:


print(TuckerAE_cluster_all[0][0][0].shape)


# In[130]:


# TuckerAE_cluster_all = (num_cluster, num_planes, num_latentdims, num_instances, 39, 39)

TuckerAE_Spectral_per_plane = np.zeros((1,4,16395,39,39))

max_f0_norm_abs = np.zeros((n_comp+1, 4))
max_f0_raw_abs = np.zeros((n_comp+1, 4))
for i in range(n_comp):
    for j in range(1):
        for k in range(len(latent_dims)):
            temp = np.array(TuckerAE_cluster_all[i][j][k])
            TuckerAE_Spectral_per_plane[j][k][index_cluster[i]] = temp
            for l in range(temp.shape[0]):
                TuckerAE_Spectral_per_plane[j][k][index_cluster[i][l]] = temp[l]
                f0_f_norm = np.copy(temp[l])
                diff = all_planes[j][index_cluster[i][l]]-f0_f_norm
                f0_abs = np.max(np.abs(diff))
                if (max_f0_norm_abs[i][k] < f0_abs):
                    max_f0_norm_abs[i][k] = f0_abs
                f0_f_raw = f0_f_norm * sig[j][index_cluster[i][l]] + mu[j][index_cluster[i][l]]
                diff = data_load[j][index_cluster[i][l]]-f0_f_raw
                f0_max = np.max(data_load[j][index_cluster[i][l]])
                f0_abs = np.max(np.abs(diff))/f0_max
                if (max_f0_raw_abs[i][k] < f0_abs):
                    max_f0_raw_abs[i][k] = f0_abs;

print ('max norm error', max_f0_norm_abs)
print ('max raw error', max_f0_raw_abs)
# In[131]:


'''
print(TuckerAE_Spectral_per_plane.shape)


# In[132]:


TuckerAE_Spectral_per_plane = np.moveaxis(TuckerAE_Spectral_per_plane, 0,1)
TuckerAE_Spectral_per_plane.shape


# In[133]:


draw(all_planes[0][11050])


# In[134]:


draw(TuckerAE_Spectral_per_plane[3][0][11050])


# In[135]:


# Denormalizing

TuckerAE_Spectral_per_plane_denorm = np.zeros((4,1,16395,39,39))

for i in range(4):
    for p in range(1):
        f0_f_raw = np.copy(TuckerAE_Spectral_per_plane[i][p])
        f0_f_raw = f0_f_raw*sig[p,:,np.newaxis, np.newaxis] + mu[p,:,np.newaxis, np.newaxis]
        TuckerAE_Spectral_per_plane_denorm[i][p] = f0_f_raw


# In[136]:


draw(data_load[0][13250])


# In[137]:


draw(TuckerAE_Spectral_per_plane_denorm[3][0][13250])


# In[138]:


print(TuckerAE_Spectral_per_plane_denorm.shape)


# In[139]:


np.save('./results/npy_files/TuckerAE_Tania_Cluster2_plane1_denorm.npy', TuckerAE_Spectral_per_plane_denorm)


# In[ ]:



'''
