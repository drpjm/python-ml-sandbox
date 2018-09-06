'''
Created on Sep 5, 2018

@author: pjmartin
'''

import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_mnist_scaled(test_size=0.2):
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target,test_size=test_size,random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # TODO: Cast targets to ints!
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':

    print("Quick demo to build a multi-layer perceptron with TF's high level API.")
    test_size = 0.25
    print("Create train/test set with test size ratio of {0}".format(test_size))
    X_train, X_test, y_train, y_test = get_mnist_scaled(test_size)
    print("Shapes...\nX_train: {0}\ny_train: {1}\nX_test: {2}\ny_train: {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    print("Building an MLP with TensorFlow...")
    # The TF DNNClassifier creates a default MLP architecture with provided layer values and softmax output layer
    features = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
#     dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=features)
#     dnn_clf_sk = tf.contrib.learn.SKCompat(dnn_clf)
#     dnn_clf_sk.fit(X_train, y_train, batch_size=50, steps=40000)
    
    
    