import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import cPickle as pickle
import gzip
from tqdm import tqdm
from Configuration import featureExtractionCNNOutputPath, featureExtractionSurfOutputPath, ClusterSize

def loadCNNData(data):
    X_labels = []
    X_features = []
    for i in range(len(data)):
        image_id = data[i][0]
        region_id = data[i][1]
        if('A' in region_id):
            region_label = 'Axon'
        elif('M' in region_id):
            region_label = 'Myelin'
        else:
            region_label = 'Schwann'
        region_features = data[i][2:]

        X_features.append(region_features)
        X_labels.append(region_label)
    # and back to the for loop

    return (X_labels, X_features)

def predictUsingCNNFeatures():

    filename = featureExtractionCNNOutputPath + "CNNFeaturesTrain.npz"
    with gzip.open(filename, 'r') as f:
        trainCNNFeature = pickle.load(f)

    (X_train_labels, X_train_features) = loadCNNData(trainCNNFeature)

    filename = featureExtractionCNNOutputPath + "CNNFeaturesTest.npz"
    with gzip.open(filename, 'r') as f:
        testCNNFeature = pickle.load(f)

    (X_test_labels, X_test_features) = loadCNNData(testCNNFeature)

    #Training SVM
    clf = svm.SVC()
    clf.fit(X_train_features, X_train_labels)
    predicted = clf.predict(X_test_features)

    print precision_recall_fscore_support(X_test_labels, predicted, average='weighted')

    print predicted

def loadSiftData(data):
    X_labels = []
    X_features = []
    for i in range(len(data)):
        if(data[i][ClusterSize]==0):
            region_label = 'Axon'
        elif(data[i][ClusterSize]==1):
            region_label = 'Myelin'
        elif(data[i][ClusterSize]==2):
            region_label = 'Schwann'
        region_features = data[i][0:ClusterSize]

        X_features.append(region_features)
        X_labels.append(region_label)
    # and back to the for loop

    return (X_labels, X_features)

def predictUsingSurfFeatures():
    filename = featureExtractionSurfOutputPath + "SiftFeaturesTrain.npz"
    with gzip.open(filename, 'r') as f:
        trainSiftFeature = pickle.load(f)

    (X_train_labels, X_train_features) = loadSiftData(trainSiftFeature)

    filename = featureExtractionSurfOutputPath + "SiftFeaturesTest.npz"
    with gzip.open(filename, 'r') as f:
        testSiftFeature = pickle.load(f)

    (X_test_labels, X_test_features) = loadSiftData(testSiftFeature)

    #Training SVM
    clf = svm.SVC()
    clf.fit(X_train_features, X_train_labels)
    predicted = clf.predict(X_test_features)

    print precision_recall_fscore_support(X_test_labels, predicted, average='weighted')

    print predicted
