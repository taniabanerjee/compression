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


import f0_diag_test
xgc = f0_diag_test.XGC_f0_diag(xgcexp)


# In[8]:


all_planes = np.copy(data_load)

# changed for separate tucker

mu = np.mean(all_planes, axis=(2,3))
sig = np.std(all_planes, axis=(2,3))
all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis]
print(all_planes.shape, mu.shape, sig.shape)


# In[84]:


Epochs = 200
Lr_Rate = 1e-4
Batch_Size = 128


# In[85]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent, e_filt, d_filt):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(1, e_filt, 3, stride=2, padding=1)
        self.enc2 = nn.LeakyReLU(negative_slope=0.2)
        self.enc3 = nn.Conv2d(e_filt, e_filt, 3, stride=2, padding=1)
        self.enc4 = nn.LeakyReLU(negative_slope=0.2)
        self.enc5 = nn.Dropout(p=0.4)
        self.enc6 = nn.Flatten()
        self.enc7 = nn.Linear(in_features=9*9*e_filt, out_features=latent)
            
        # Decoder
        self.dec1 = nn.Linear(in_features=latent, out_features=9*9*d_filt)
        self.dec2 = nn.LeakyReLU(negative_slope=0.2)
        self.dec3 = nn.ConvTranspose2d(d_filt, d_filt, 2, stride=2)
        self.dec4 = nn.LeakyReLU(negative_slope=0.2)
        self.dec5 = nn.ConvTranspose2d(d_filt, d_filt, 2, stride=2)
        self.dec6 = nn.LeakyReLU(negative_slope=0.2)
        self.dec7 = nn.Conv2d(d_filt, 1, kernel_size=9, padding=4)
        

    def encode(self, x):
        x = torch.reshape(x, (x.size(0),1,36,36))
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = self.dec2(x)
        x = torch.reshape(x, (x.size(0),d_filt,9,9))
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = torch.reshape(x, (x.size(0),1,36*36))
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# In[5]:


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# In[86]:


from torchsummary import summary
#latent_dims = [32, 64, 128, 256]
latent_dims = [128]
e_filt=256
d_filt=256
for latent in latent_dims:
    model = Autoencoder(1296, latent, e_filt=256, d_filt=256)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
    device = get_device()
    model.to(device)
    print(summary(model, (1, 1296)))


# In[88]:


def training(model, train_loader, Epochs):
    train_loss = []
    best_loss = 100
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
        
        if (loss < best_loss):
            torch.save(model.state_dict(), './model/best-model-parameters-conv.pt')
            best_loss = loss
            print('best loss: ', best_loss)

    return train_loss


# In[89]:


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


# In[90]:


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


# In[91]:


def decoding(model, test_loader, num_instance, input_dim):
    #reconstructions = np.zeros((78*16,num_coeffs))
    reconstructions = []    

    for batch in test_loader:
        data = batch[0]
        data = data.to(device)
        outputs = model.decode(data)
        outputs_cpu = outputs.cpu().data.numpy()
        outputs_cpu = np.reshape(outputs_cpu, (outputs_cpu.shape[0],input_dim))
        reconstructions.append(outputs_cpu)
        #reconstructions[count] = outputs.cpu().numpy()
        #count = count + 1
    
    decoded_vectors = np.zeros((num_instance,input_dim), dtype=np.float32)
    count = 0
    for i in range(len(reconstructions)):
        for j in range(len(reconstructions[i])):
            decoded_vectors[count] = np.float32(reconstructions[i][j])
            count = count + 1
    
    return decoded_vectors


# In[92]:


def rmse_error(x, y):

    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)

    return np.sqrt(mse)


# In[93]:


def physics_loss_rmse(data, data_recon):
    batch_size = data.shape[0]

    f0_f = np.copy(data)
    den, u_para, T_perp, T_para, _, _ = xgcexp.f0_diag(0, batch_size, 1, f0_f)
        
    f0_f = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon, _, _ = xgcexp.f0_diag(0, batch_size, 1, f0_f)
    
    den_err = rmse_error(den_recon, den)
    u_para_err = rmse_error(u_para_recon, u_para)
    T_perp_err = rmse_error(T_perp_recon, T_perp)
    T_para_err = rmse_error(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[94]:


def compute_diff(x, x_):
    assert(x.shape == x_.shape)
    max_diff = np.max(np.abs(x - x_))
    max_v = np.max(np.abs(x))
    
    return max_diff/max_v


# In[95]:


def physics_loss_Linf(data, data_recon):
    batch_size = data.shape[0]
    
    den = np.zeros((batch_size))
    u_para = np.zeros((batch_size))
    T_perp = np.zeros((batch_size))
    T_para = np.zeros((batch_size))
    
    den_recon = np.zeros((batch_size))
    u_para_recon = np.zeros((batch_size))
    T_perp_recon = np.zeros((batch_size))
    T_para_recon = np.zeros((batch_size))

    f0_f0 = np.copy(data)
    den, u_para, T_perp, T_para = xgc.f0_diag(isp=1, f0_f=f0_f0)
    f0_f0 = np.copy(data_recon)
    den_recon, u_para_recon, T_perp_recon, T_para_recon = xgc.f0_diag(isp=1, f0_f=f0_f0)
    
    #den_err = compute_diff(den_recon, den)
    #u_para_err = compute_diff(u_para_recon, u_para)
    #T_perp_err = compute_diff(T_perp_recon, T_perp)
    #T_para_err = compute_diff(T_para_recon, T_para)
    
    den_err = rmse_error(den_recon, den)
    u_para_err = rmse_error(u_para_recon, u_para)
    T_perp_err = rmse_error(T_perp_recon, T_perp)
    T_para_err = rmse_error(T_para_recon, T_para)

    return (den_err, u_para_err, T_perp_err, T_para_err)


# In[96]:


# Tucker AE - Per plane, Full Tucker coeffs

num_channels = 1

latent_dims = [32, 64, 128, 256]
e_filt = 256
d_filt = 256

TuckerAE_all = np.zeros((8,len(latent_dims),16395,39,39))
print('TuckerAE_all: ', TuckerAE_all.shape)

#for p in range(all_planes.shape[0]):
for p in range(1):
    
    X_train = all_planes[p]
    
    train_num = X_train.shape[0]
    
    results = []

    tucker_data = np.array(X_train).reshape([X_train.shape[0],39,39])
    
    tqdm.write('converting data to tucker basis...')
    coeff_tucker, factors = tucker(tucker_data)

    experiments = [
        {
            "num_coeffs": 1296
        }
    ]    

    for experiment in experiments:
        num_coeffs = experiment['num_coeffs']
        
        tqdm.write('tucker_coeffs: '+str(num_coeffs))

        top_coeffs = tucker_top_coeffs(coeff_tucker, num_coeffs)

        X_train_tucker = to_tucker_basis(np.array(X_train).reshape([X_train.shape[0],39,39]), factors, top_coeffs)
        
        print(X_train.shape)
        print(X_train_tucker.shape)
        
        X_train_tucker1 = np.reshape(X_train_tucker, (X_train_tucker.shape[0],1,X_train_tucker.shape[1]))        
        print(X_train_tucker1.shape)
        
        #X_train_tucker = X_train_tucker.reshape([700,16,num_coeffs])
        #X_test_tucker = X_test_tucker.reshape([78,16,num_coeffs])
        
        training_data_tucker = torch.utils.data.TensorDataset(torch.Tensor(X_train_tucker1))
        
        training_loader_tucker = DataLoader(training_data_tucker,
                                            batch_size=Batch_Size,
                                            shuffle=False,
                                            pin_memory=True)
        
        #train_physics_err_all = []
        #train_f0f_err_all = []
        
        for j in range(len(latent_dims)):
            latent_dim = latent_dims[j]
            model = Autoencoder(num_coeffs, latent_dim, e_filt, d_filt)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
            #optimizer = optim.RMSprop(model.parameters(), lr=Lr_Rate)
            device = get_device()
            model.to(device)
            print(summary(model, (1, 1296)))
            print(model)
            
            train_loss = training(model, training_loader_tucker, Epochs)
            
            model.load_state_dict(torch.load('./model/best-model-parameters-conv.pt'))
            model.eval()
        
            latent_train = encoding(model, training_loader_tucker, X_train.shape[0], latent_dim)
            print('latent_train shape: ', latent_train.shape)
        
            #print('latent train len: ', len(latent_space_train))
            #print('latent test len: ', len(latent_space_test))
        
            '''
            def train_quantizer():
                seed = np.random.randint(100)
                pq = nanopq.PQ(M=latent_dim, Ks=256, verbose=True)
                pq.fit(vecs=latent_train, iter=20, seed=seed)
                return pq

            pq = train_quantizer()
            latent_train_quan = pq.encode(vecs=latent_train)
        
            latent_train_dequan = pq.decode(codes=latent_train_quan)    
            '''
            
            latent_train_dequan = latent_train
            print('latent_train_dequan shape: ', latent_train_dequan.shape) 
            
            training_latent = torch.utils.data.TensorDataset(torch.Tensor(latent_train_dequan))
        
            training_loader_latent = DataLoader(training_latent,
                                                batch_size=Batch_Size,
                                                shuffle=False,
                                                pin_memory=True)
            
            recon_train_tucker = decoding(model, training_loader_latent, X_train.shape[0], num_coeffs)        
            print('recon train shape: ', recon_train_tucker.shape)
            
            recon_train_tucker1 = np.reshape(recon_train_tucker, (recon_train_tucker.shape[0],num_coeffs))
            print('recon train1 shape: ', recon_train_tucker1.shape)
        
            train_recon = from_tucker_basis(recon_train_tucker1, factors, top_coeffs, data_shape=[X_train.shape[0],39,39])
            print('train_recon shape: ', train_recon.shape)
            #train_orig_images = np.array(X_train).reshape([X_train.shape[0], 39,39])
            TuckerAE_all[p][j] = train_recon
            
            #train_physics_err = physics_Linf(train_orig_images, mu, sig, train_recon)
            #train_physics_err_all.append(train_physics_err)


# In[97]:


TuckerAE_all.shape


# In[98]:


draw(all_planes[0][11050])


# In[101]:


draw(TuckerAE_all[0][3][11050])


# In[82]:


conv_plane1 = TuckerAE_all[0]
print(conv_plane1.shape)
conv_plane1 = np.reshape(conv_plane1, (4,1,16395,39,39))
print(conv_plane1.shape)


# In[83]:


np.save('./results/npy_files/TuckerAE_1296_conv_plane1_pytorchver.npy', conv_plane1)


# In[42]:


# Denormalizing

TuckerAE_all_denorm = np.zeros((8,4,16395,39,39))

for p in range(8):
    for i in range(4):
        f0_f_raw = np.copy(TuckerAE_all[p][i])
        f0_f_raw = f0_f_raw*sig[p,:,np.newaxis, np.newaxis] + mu[p,:,np.newaxis, np.newaxis]
        TuckerAE_all_denorm[p][i] = f0_f_raw


# In[43]:


TuckerAE_all_denorm = np.moveaxis(TuckerAE_all_denorm, 0, 1)
print(TuckerAE_all_denorm.shape)


# In[47]:


np.save('./results/npy_files/TuckerAE_1296_conv_per_plane_denorm.npy', TuckerAE_all_denorm)


# In[46]:


draw(data_load[7][11050])


# In[45]:


draw(TuckerAE_all_denorm[3][7][11050])


# In[22]:


# Tucker AE - Full Tucker coeffs - whole planes

num_channels = 1

latent_dims = [32, 64, 128, 256]

TuckerAE_8planes_together_all = np.zeros((len(latent_dims),8,16395,39,39))
print('TuckerAE_8planes_together_all: ', TuckerAE_8planes_together_all.shape)

for p in range(1):
    
    X_train = np.reshape(all_planes, (8*16395,39,39))
    
    train_num = X_train.shape[0]
    
    results = []

    tucker_data = np.array(X_train).reshape([X_train.shape[0],39,39])
    
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
        
        #X_train_tucker = X_train_tucker.reshape([700,16,num_coeffs])
        #X_test_tucker = X_test_tucker.reshape([78,16,num_coeffs])
        
        training_data_tucker = torch.utils.data.TensorDataset(torch.Tensor(X_train_tucker))
        
        training_loader_tucker = DataLoader(training_data_tucker,
                                            batch_size=Batch_Size,
                                            shuffle=False,
                                            pin_memory=True)
        
        #train_physics_err_all = []
        #train_f0f_err_all = []
        
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
        
            #print('latent train len: ', len(latent_space_train))
            #print('latent test len: ', len(latent_space_test))
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
            TuckerAE_8planes_together_all[j] = np.reshape(train_recon, (8,16395,39,39))
            
            #train_physics_err = physics_Linf(train_orig_images, mu, sig, train_recon)
            #train_physics_err_all.append(train_physics_err)


# In[25]:


TuckerAE_8planes_together_all.shape


# In[24]:


np.save('./npy_files/TuckerAE_8planes_together_all.npy', TuckerAE_8planes_together_all)


# In[26]:


TuckerAE_8planes_together_all = np.load('./npy_files/TuckerAE_8planes_together_all.npy')
print(TuckerAE_8planes_together_all.shape)


# In[27]:


# Denormalizing

TuckerAE_8planes_together_all_denorm = np.zeros((4,8,16395,39,39))

for i in range(4):
    for p in range(8):
        f0_f_raw = np.copy(TuckerAE_8planes_together_all[i][p])
        f0_f_raw = f0_f_raw*sig[p,:,np.newaxis, np.newaxis] + mu[p,:,np.newaxis, np.newaxis]
        TuckerAE_8planes_together_all_denorm[i][p] = f0_f_raw


# In[44]:


TuckerAE_8planes_together_all_denorm.shape


# In[34]:


np.save('./npy_files/TuckerAE_8planes_together_denorm.npy', TuckerAE_8planes_together_all_denorm)


# In[39]:


draw(data_load[7][13050])


# In[40]:


draw(TuckerAE_8planes_together_all_denorm[3][7][13050])


# In[ ]:




