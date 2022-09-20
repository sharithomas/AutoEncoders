
# 7 digit Mnist _Creation for 50k samples


import copy

import glob

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np 
import random 
import cv2
import pandas as pd

list_df=[]
df=pd.DataFrame()
for k in range(0,50000):
    #f = open(path_to_file, "w")

    def get_random_MNIST_digit(digit_label ):
        #get a random image with provided label    
        train_index = np.where(y_train == digit_label ) #get all y trains with given label
        random_index = np.random.choice(train_index[0], size=1) #pick a random index 
        digit = X_train[random_index][0,:,:] #get the 28x28 pixels of random index 
        return digit
     
                                
    def display_img(img ):
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
    def create_multidigit_img (randomlist) :    
        final_image= np.zeros(shape=(28, 1))
        image_name =""
        for i in randomlist : 
            single_digit =get_random_MNIST_digit(i)
            single_digit_padded = add_remove_columns(single_digit)        
            final_image= np.hstack((final_image,single_digit_padded))#combine arrays horizontally
            image_name = image_name+ str(i) 
        return final_image , image_name
    
    def add_remove_columns (img): 
        #get flag 0 or 1 randonly . if 0 => remove columns , if 1 => add columns
        number_of_columns = 4 #number of columns to be added or removed from both ends of MNIST image
        
        #if random.sample(range(0, 2),1)[0] == 0 :         
        altered_image = img[: , 4:-4]        
            
        # else:    
        #     number_of_columns = 5
        #     padding_image = np.zeros(shape=(28, number_of_columns))
            
        #     altered_image = np.hstack((padding_image ,img)) #add 5 columns at front
        #     altered_image = np.hstack((altered_image,padding_image))#add 5 columns at end        
      
        return altered_image
     
            
        
    
    # load dataset
    (X_train, y_train) , (X_test, y_test) = mnist.load_data()
    number_of_digits = 7
    randomlist =[]
    for i in range (0,number_of_digits) :
        random_number = random.sample(range(0, 9),1)[0] #random.sample gives list format- take 0 index
        randomlist.append(random_number)
        
    final_image ,image_name =  create_multidigit_img (randomlist)  
