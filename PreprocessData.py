import pandas as pd
import cPickle as pickle
import gzip
from Configuration import trainFile, testFile, preprocessOutputPath

def preProcessRegions(isTrain):
    if(isTrain):
        filename = trainFile
    else:
        filename = testFile
    regions = []
    regdata = pd.read_csv(filename, sep='~', names=['ImgId', 'RegId', 'RegLabel', 'RX', 'RY', 'RW', 'RH', 'IW', 'IH'])

    regdata['image_id'] = regdata['ImgId']
    regdata['region_id'] = regdata['RegId'].apply(lambda x: x.split('.')[0])
    regdata['region_label'] = regdata['RegLabel']
    regdata['region_X'] = regdata['RX']
    regdata['region_Y'] = regdata['RY']
    regdata['region_W'] = regdata['RW']
    regdata['region_H'] = regdata['RH']
    regdata['image_W'] = regdata['IW']
    regdata['image_H'] = regdata['IH']

    regdata = regdata[['image_id', 'region_id', 'region_label', 'region_X', 'region_Y', 'region_W', 'region_H', 'image_W', 'image_H']]

    if(isTrain):
        outputFile = preprocessOutputPath + 'regions_train.pklz'
    else:
        outputFile = preprocessOutputPath + 'regions_test.pklz'

    with gzip.open(outputFile, 'w') as f:
        pickle.dump(regdata, f)