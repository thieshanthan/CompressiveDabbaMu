import torch
import torchvision
import shutil 
import matplotlib.pyplot as plt
import numpy as np
import os, shutil


mean = 0
std=1

img_size=32
torch.manual_seed(10)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([img_size, img_size]),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                 (mean,), (std,))
                             ])),
  batch_size=60000, shuffle=True)

val_test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([img_size, img_size]),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                 (mean,), (std,))
                             ])),
  batch_size=10000, shuffle=True, drop_last= True)



train_data, _ =next(iter(train_loader))
val_test_data, _ =next(iter(val_test_loader))

## CORRECT WAY TO SHUFFLE WITHOUT REPEATS !!!
#val_test_data = val_test_data[np.random.choice(np.arange(len(val_test_data)), len(val_test_data), replace=False)] ## SHUFFLE !!!
u, c = np.unique(val_test_data, return_index=True, axis=0)
print(f'number of total samples: {len(val_test_data)}')
print(f'unique : {u.shape}')

def save_grids(data, images_per_grid, nrow_per_grid, start_idx, save_dir, type_, remove_existing_dir= False):
    if remove_existing_dir:
        try:shutil.rmtree(f"{save_dir}/{type_}")
        except:pass
    
    try:os.mkdir(f"{save_dir}/{type_}")
    except:pass
    
    np.random.seed(500)
    data = data[np.random.choice(np.arange(len(data)), len(data), replace=False)] ## SHUFFLE !!!
    
    ## CHECK DUPLICATES
    uniques, c = np.unique(data, return_index=True, axis=0)
    assert uniques.shape==data.shape, f'DATASET HAVE DUPLICATED SAMPLES !!! --> uniques.shape ({uniques.shape}) != data.shape ({data.shape})'
    print(f'NO DUPLICATES in the dataset ...  --> uniques.shape ({uniques.shape}) == data.shape ({data.shape})')
    ## CHECK DUPLICATES

    for i in range(0, len(data), images_per_grid):
        grid = torchvision.utils.make_grid(data[i:i+images_per_grid], padding=0, nrow= nrow_per_grid).permute(1,2,0).cpu().numpy()
        
        img_idx= start_idx + i//images_per_grid+1
        img_save_dir = f"{save_dir}/{type_}/{img_idx}.jpg"
        plt.imsave(img_save_dir, grid)
        if i==0: print("saving ...  (first grid image): ", img_save_dir)
    print(f"saving ...  (last grid image): {img_save_dir}\n")
    
    
################################################### CREATE DATASET

images_per_grid= 100*4
nrow_per_grid= int(images_per_grid**0.5)

repeat_train= 100*4
repeat_valtest= 84

save_dir = f"./datasets/mnistgrid_mnistsize({img_size})_imgsize({img_size*nrow_per_grid})"
try:os.mkdir('./datasets')
except:pass

print('dataset directory :: ', save_dir)

try:os.mkdir(save_dir)
except:print(f'available directory :: {save_dir}')

for idx in range(repeat_train):
    if idx==0:remove_existing_dir= True
    else:remove_existing_dir= False
        
    save_grids(train_data, images_per_grid, nrow_per_grid, idx*(len(train_data)//images_per_grid), save_dir, 'train', remove_existing_dir= remove_existing_dir) # save 600 images
    
for idx in range(repeat_valtest):#range(20*4):
    if idx==0:remove_existing_dir= True
    else:remove_existing_dir= False
    
    val_test_data = val_test_data[np.random.choice(np.arange(len(val_test_data)), len(val_test_data), replace=False)] ## SHUFFLE WITHOUT REPEATS!!!
    save_grids(val_test_data[:5000], images_per_grid, nrow_per_grid, idx*(5000//images_per_grid), save_dir, 'val', remove_existing_dir= remove_existing_dir) # save 50 images
    save_grids(val_test_data[5000:], images_per_grid, nrow_per_grid, idx*(5000//images_per_grid), save_dir, 'test', remove_existing_dir= remove_existing_dir) # save 50 images