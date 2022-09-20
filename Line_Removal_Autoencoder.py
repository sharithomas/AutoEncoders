
# import libraries
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import copy

import glob

from scipy.ndimage.measurements import center_of_mass 
import math

# use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Standard image Size
height =28
width=256

#Image Padding

def image_padding(image):
    (h, w) = image.shape[:2]
    #padding on image
    colsPadding = (int(math.ceil((width-w)/2.0)),int(math.floor((width-w)/2.0)))
    rowsPadding = (int(math.ceil((height-h)/2.0)),int(math.floor((height-h)/2.0)))

    
    image_padded = np.lib.pad(image,(rowsPadding,colsPadding),'constant',constant_values=(255,255))
    #plt.imshow(image_padded)
    
    
    cy,cx = center_of_mass(image_padded)

    rows,cols = image_padded.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    
 
    cy,cx = center_of_mass(image_padded)

    rows,cols = image_padded.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    
    rows,cols = image_padded.shape
    M = np.float32([[1,0,shiftx],[0,1,shifty]])
    shifted = cv2.warpAffine(image_padded,M,(cols,rows))
    
    return shifted
    
 #Image Resize without Diatortion
 def resize_without_distortion(img):
  def image_resize(image, width, height , inter = cv2.INTER_AREA):
    
    dim=None
    (h, w) = image.shape[:2]
    
    # 1. return original image
    if width is w and height is h:
      print("Same image")
      return image
   
    #2.image is small
    elif w<width or  h<height:
        
      wt_ratio=width/float(w)
      ht_ratio=height/float(h)
        
      #too small
      if wt_ratio >=2 and ht_ratio>=2:
        if wt_ratio<ht_ratio:
          #print("Too small 1 case")
          dim=(int(w * wt_ratio), int(h * wt_ratio))
          resized = cv2.resize(image, dim)
                
          #padding on height 
          final_image=image_padding(resized)
          #plt.imshow(final_image)
          return final_image
         
        else:
          #print("Too small 2 case")
          dim=dim = (int(w * ht_ratio), int(h * ht_ratio))
          resized = cv2.resize(image, dim)
                
          #padding on width
          final_image=image_padding(resized)
          #plt.imshow(final_image)
          return final_image
      else:
        if w<width and h<height:
            final_image=image_padding(image)
            return final_image
        elif w<width and h>height:
            dim=(w, height)
            resized = cv2.resize(image, dim)
            print(resized.shape)
            final_image=image_padding(resized)
            return final_image
        else:
            dim=(width, h)
            resized = cv2.resize(image, dim)
            final_image=image_padding(resized)
            return final_image
    else:
        wt_ratio=w/float(width)
        ht_ratio=h/float(height)
        if wt_ratio>ht_ratio:
            #print("Large image 1 case")
            dim=(int(w /float(wt_ratio)), int(h / float(wt_ratio)))
            resized = cv2.resize(image, dim)
            final_image=image_padding(resized)
            return final_image
            #plt.imshow(final_image)
        else:
            dim=(int(w /float(ht_ratio)), int(h / float(ht_ratio)))
            resized = cv2.resize(image, dim)
            final_image=image_padding(resized)
            return final_image

  result_image= image_resize(img, width, height , inter = cv2.INTER_AREA)
  return result_image
  
  !unzip /content/drive/MyDrive/HWR/dataset.zip
  
  filelist = glob.glob('/content/dataset/*.png')
  
  img_count=len(filelist)
  
  #Create Tensor images with and without Ocllusion
  def image_preprocessing(filelist):
    i=0
    imagesOcc   = torch.zeros(img_count,1,28,256)
    imagesNoOcc = torch.zeros(img_count,1,28,256)
    for fname in filelist:
        #print(fname)
        img=cv2.imread(fname)
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        gray_image=255-gray_image

        #img_resize = cv2.resize(255-gray_image, (224,56)) 
        Noocclusion_resized=resize_without_distortion(gray_image)
        #occluded_resized=resize_without_distortion(occluded)

        occluded = copy.deepcopy( gray_image )
        i1 = np.random.choice(np.arange(10,18))
        i2 = np.random.choice(np.arange(2,3))
        occluded[i1:i1+i2,] = 0

        occluded_resized=resize_without_distortion(occluded)
        
        NoOcc_dataNorm = Noocclusion_resized / np.max(Noocclusion_resized)
        Occ_dataNorm = occluded_resized / np.max(occluded_resized)


        # convert to tensor
        NoOcc_dataT= torch.tensor( NoOcc_dataNorm ).float()
        Occ_dataT= torch.tensor( Occ_dataNorm ).float()
        
        #img=NoOcc_dataT
        
        imagesNoOcc[i,:,:,:]=torch.Tensor(NoOcc_dataT).view(1,28,256)
    
        imagesOcc[i,:,:,:] = torch.Tensor(Occ_dataT).view(1,28,256)
        i=i+1
    return imagesNoOcc,imagesOcc

imagesNoOcc,imagesOcc=image_preprocessing(filelist)

#Display of few sample images 
fig,ax = plt.subplots(2,5,figsize=(15,3))

for i in range(5):
    
  whichpic = np.random.randint(img_count)
    
  I1 = torch.squeeze(imagesNoOcc[whichpic,:,:] ).detach()
  I2 = torch.squeeze( imagesOcc[whichpic,:,:] ).detach()
  
    
  ax[0,i].imshow(I1)
  ax[0,i].set_xticks([]), ax[0,i].set_yticks([])
  
  ax[1,i].imshow(I2 )
  ax[1,i].set_xticks([]), ax[1,i].set_yticks([])

plt.show()

#Model Architecture

# create a class for the model
def makeTheNet():

  class gausnet(nn.Module):
    def __init__(self):
      super().__init__()
      
      # encoding layer
      self.enc = nn.Sequential(
          nn.Conv2d(1, 16, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(32, 64, 7),

   
          )
      
      # decoding layer
      self.dec = nn.Sequential(
          nn.ConvTranspose2d(64, 32, 7),
          nn.ReLU(),
          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
          nn.Sigmoid()
          )
      
    def forward(self,x):
      return self.dec( self.enc(x) )
  
  # create the model instance
  net = gausnet()
  
  # loss function
  lossfun = nn.MSELoss()

  # optimizer
  optimizer = torch.optim.Adam(net.parameters(),lr=.001)

  return net,lossfun,optimizer

# test the model with one batch
net,lossfun,optimizer = makeTheNet()

yHat = net(imagesOcc[:5,:,:,:])

# check size of output
print(' ')
print(yHat.shape)
print(imagesOcc.shape)


# let's see how they look
fig,ax = plt.subplots(1,2,figsize=(8,3))
ax[0].imshow(torch.squeeze(imagesOcc[0,0,:,:]).detach(),cmap='jet')
ax[0].set_title('Model input')
ax[1].imshow(torch.squeeze(yHat[0,0,:,:]).detach(),cmap='jet')
ax[1].set_title('Model output')

plt.show()


# a function that trains the model

def function2trainTheModel():

  # number of epochs
  numepochs = 1000
  
  # create a new model
  net,lossfun,optimizer = makeTheNet()

   # model
  net.to(device)


  # initialize losses
  losses = torch.zeros(numepochs)

  # loop over epochs
  for epochi in range(numepochs):
      
    print("Epoch no: ",epochi)
      
    # pick a set of images at random
    pics2use = np.random.choice(img_count,size=32,replace=False)

    # get the input (has occlusions) and the target (no occlusions)
    X = imagesOcc[pics2use,:,:,:]
    Y = imagesNoOcc[pics2use,:,:,:]

    # data
    X=X.to(device)
    Y=Y.to(device)

    # forward pass and loss
    yHat = net(X)
    loss = lossfun(yHat,Y)
    losses[epochi] = loss.item()
    

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # end epochs

  # function output
  return losses,net,Y,yHat

# test the model on a bit of data
losses,net,Y,yHat = function2trainTheModel()

file='/content/drive/MyDrive/HWR/model.pth'
torch.save(net.state_dict(), file)

plt.plot(losses,'s-',label='Train')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Model loss')

plt.show()

#Output for test images
pics2use = np.random.choice(img_count,size=32,replace=False)
X = imagesOcc[pics2use,:,:,:].to(device)
yHat = net(X)

fig,ax = plt.subplots(2,5,figsize=(15,3))
for i in range(5):
    
  #whichpic = np.random.randint(img_count)
    
  I1 = torch.squeeze(X[i,:,:] ).detach().cpu()
  I2 = torch.squeeze( yHat[i,:,:] ).detach().cpu()
  
    
  ax[0,i].imshow(I1)
  ax[0,i].set_xticks([]), ax[0,i].set_yticks([])
  
  ax[1,i].imshow(I2 )
  ax[1,i].set_xticks([]), ax[1,i].set_yticks([])

plt.show()

#Inference Model
net.load_state_dict(torch.load('/content/drive/MyDrive/HWR/model.pth'))

image=cv2.imread('/content/drive/MyDrive/HWR/sample.png')

plt.imshow(image)

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_image=255-gray_image

resized=resize_without_distortion(gray_image)

dataNorm = resized / np.max(resized)

 # convert to tensor
dataT= torch.tensor( dataNorm ).float()

images_inference = torch.zeros(1,1,28,256)

images_inference[0,:,:,:]=torch.Tensor(dataT).view(1,28,256)
images_inference=images_inference.to(device)

yHat_inference = net(images_inference).to(device)
plt.imshow(torch.squeeze( images_inference[0,:,:] ).cpu())

plt.imshow(torch.squeeze( yHat_inference[0,:,:] ).detach().cpu())



