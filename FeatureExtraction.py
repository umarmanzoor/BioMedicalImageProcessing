import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cPickle as pickle
import gzip
import cv2
from Configuration import imageDir, featureExtractionOutputPath, showImage, useCNNFeatures
from PIL import Image as PImage
from sklearn.cluster import KMeans

# GoogleNET CNN
from sklearn_theano.feature_extraction import GoogLeNetTransformer
from sklearn_theano.feature_extraction.caffe.googlenet_layer_names import get_googlenet_layer_names

from sklearn.metrics import precision_recall_fscore_support

from sklearn import svm

def computeCNNFeatures(regions, filename):

    # extract CNN features for regions
    gltr = GoogLeNetTransformer(force_reshape=True,
                                batch_size=10,
                                output_layers=(get_googlenet_layer_names()[-4],))
    X_i = []
    ids = []
    for n, row in tqdm(regions.iterrows(), total=len(regions)):
        this_image_id = row['image_id']
        this_region_id = row['region_id']
        this_bb = [int(row['region_X']), int(row['region_Y']), int(row['region_W']), int(row['region_H'])]
        this_image_dim = [int(row['image_W']), int(row['image_H'])]
        (region_cropped, kp, desc) = getCroppedRegion(this_image_id, this_bb)
        X_i.append(region_cropped)
        ids.append(np.array([this_image_id, this_region_id]))
    # and back to the for loop

    filenamePath = featureExtractionOutputPath + filename
    # Write features to file
    print "Writing features to file...!"
    X = gltr.transform(X_i)
    X_ids = np.array(ids)
    X_f = np.hstack([X_ids,X])

    np.savez_compressed(filenamePath, X_f)
    with gzip.open(filenamePath, 'w') as f:
        pickle.dump(X_f, f)
    print X_f.shape

def getCroppedRegion(imageId, bb):
    imagePath = imageDir + imageId
    img = cv2.imread(imagePath)
    if len(img.shape) > 2 and img.shape[2] == 4:
        # convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    x,y,w,h = np.clip(np.array(bb), 0, np.max(img.shape))
    w = img.shape[1]-x if x+w >= img.shape[1] else w
    h = img.shape[0]-y if y+h >= img.shape[0] else h
    regionCropped = np.array(img[y:y+h,x:x+w])
    # Surf key points and desc
    kp, desc = gen_sift_features(regionCropped)
    regionCropped = np.array(img[y:y+h,x:x+w])
    if (useCNNFeatures):
        xs = 224
        ys = 224
        pim = PImage.fromarray(regionCropped)
        pim2 = pim.resize((xs,ys), PImage.ANTIALIAS)
        regionCroppedCNN = np.array(pim2)
    else:
        # Surf key points and desc
        kp, desc = gen_sift_features(regionCropped)

    if(showImage):
        plt.imshow(cv2.drawKeypoints(regionCropped, kp, None, color=(0, 255, 0), flags=0))
        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(img)
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()
    if(useCNNFeatures):
        return (regionCroppedCNN, [], [])
    else:
        return (regionCropped, kp, desc)

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def show_rgb_img(img):
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def computeSurfFeatures(regions):
    croppedRegions = []
    imageIds = []
    regionIds = []
    regionlabels = []
    keypoints = []
    keypointsDesc = []
    allImageKP = []
    allImageKPDesc = []
    print '-' * 40
    print "Calculating Surf KeyPoints..."
    for n, row in tqdm(regions.iterrows(), total=len(regions)):
        this_image_id = row['image_id']
        this_region_id = row['region_id']
        this_region_label = row['region_label']
        this_bb = [int(row['region_X']), int(row['region_Y']), int(row['region_W']), int(row['region_H'])]
        this_image_dim = [int(row['image_W']), int(row['image_H'])]
        (rC, kp, desc) = getCroppedRegion(this_image_id, this_bb)
        croppedRegions.append(rC)
        imageIds.append(this_image_id)
        regionIds.append(this_region_id)
        regionlabels.append(this_region_label)
        keypoints.append(kp)
        if len(kp)>0:
            keypointsDesc.append(desc)
        else:
            keypointsDesc.append([])
        # Append Keypoint and desc
        for k in range(len(kp)):
            allImageKP.append(kp[k])
            allImageKPDesc.append(desc[k])
    # and back to the for loop

    print '-' * 40
    print "Generating Clusters..."
    # KMean
    # Number of clusters
    kmeans = KMeans(n_clusters=100)
    # Fitting the input data
    kmeans = kmeans.fit(allImageKPDesc)
    # Getting the cluster labels
    labels = kmeans.predict(allImageKPDesc)
    # Centroid values
    centroids = kmeans.cluster_centers_

    #Feature Vector Generation
    print '-' * 40
    print "Generating Feature Vector..."
    features = []
    count = 0
    for i in range(len(keypointsDesc)):
        if len(keypointsDesc[i]) > 0:
            indexes = kmeans.predict(keypointsDesc[i])
            kpFeature = [0] * 100
            for j in range(len(indexes)):
                kpFeature[indexes[j]] = kpFeature[indexes[j]] + 1
            features.append(kpFeature)
        else:
            kpFeature = [0] * 100
            features.append(kpFeature)
            count += 1
            print i

    print count
    #Training SVM
    clf = svm.SVC()
    clf.fit(features, regionlabels)

    predicted = clf.predict(features)

    print precision_recall_fscore_support(regionlabels, predicted, average='weighted')

    print predicted