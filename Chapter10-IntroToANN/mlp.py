'''
Created on Sep 5, 2018

@author: pjmartin
'''

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Note that TF has moved *a lot* of its example apis around. This code merges
# common practice with sklearn and the new tf.estimator organization. 

def mnist_to_datasets(test_size=0.2):
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target,test_size=test_size,random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_dataset = tf.data.Dataset.from_tensor_slices( (X_train_scaled, y_train) )
    test_dataset = tf.data.Dataset.from_tensor_slices( (X_test_scaled, y_test) )
    return {'train': train_dataset, 'test' : test_dataset}



if __name__ == '__main__':

    dataset_dict = mnist_to_datasets(0.25)

    
    
    
    