from time import strftime
import cPickle as pickle
import gzip

import numpy as np
from tqdm import tqdm

from PreprocessData import preProcessRegions
from Configuration import preprocess, preprocessOutputPath, useCNNFeatures, computeFeature

from FeatureExtraction import computeCNNFeatures, computeSurfFeatures
from DataClassification import loadRegionFeatures

if __name__ == '__main__':

    # preprocess regions data
    if(preprocess):
        print 'Preprocessing Data...'
        preProcessRegions(True)
        preProcessRegions(False)
        print 'Preprocessing Complete...'

    if(computeFeature):
        #load regions
        trainPath = preprocessOutputPath + 'regions_train.pklz'
        with gzip.open(trainPath, 'r') as f:
            trainRegions = pickle.load(f)

        testPath = preprocessOutputPath + 'regions_test.pklz'
        with gzip.open(testPath, 'r') as f:
            testRegions = pickle.load(f)

        print '-' * 40
        print strftime("%Y-%m-%d %H:%M:%S")
        print "Extraction Starts..."

        if (useCNNFeatures):
            computeCNNFeatures(trainRegions, "CNNFeaturesTrain.npz")
            computeCNNFeatures(testRegions, "CNNFeaturesTest.npz")
        else:
            # OpenCv Sift Features
            computeSurfFeatures(trainRegions)

        print '-' * 40
        print 'Extraction Complete...'
        print strftime("%Y-%m-%d %H:%M:%S")

    loadRegionFeatures()