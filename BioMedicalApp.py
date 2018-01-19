from time import strftime
import cPickle as pickle
import gzip

from PreprocessData import preProcessRegions
from Configuration import preprocess, preprocessOutputPath, useCNNFeatures

from FeatureExtraction import computeCNNFeatures, computeSurfFeatures

if __name__ == '__main__':

    # preprocess regions data
    if(preprocess):
        print 'Preprocessing Data...'
        preProcessRegions(True)
        #preProcessRegions(False)
        print 'Preprocessing Complete...'

    #load regions
    trainPath = preprocessOutputPath + 'regions_train.pklz'
    with gzip.open(trainPath, 'r') as f:
        trainRegions = pickle.load(f)

    # testRegions = preprocessOutputPath + 'regions_test.pklz'
    # with gzip.open(testRegions, 'r') as f:
    #     regionsTest = pickle.load(f)

    print '-' * 40
    print strftime("%Y-%m-%d %H:%M:%S")
    print "Extraction Starts..."

    if (useCNNFeatures):
        computeCNNFeatures(trainRegions)
        #concatenate_feat_batches(trainRegions, testRegions)
    else:
        # OpenCv Sift Features
        computeSurfFeatures(trainRegions)

    print '-' * 40
    print 'Extraction Complete...'
    print strftime("%Y-%m-%d %H:%M:%S")