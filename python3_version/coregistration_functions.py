# Copyright (c) 2012, Jan Erik Solem
# All rights reserved.
#
# Copyright (c) 2019, Anette Eltner
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os, csv
import numpy as np
import pylab as plt
import matplotlib
import pandas as pd

import cv2

import draw_functions as drawF
import featureTracking_functions as trackF


'''perform coregistration'''
def coregistration(image_list, directory_out, reduceKepoints=None, descriptorVersion="orb",
                   feature_match_twosided=False, nbr_good_matches=10,
                   master_0 = True):
# descriptor versions: sift, orb, akaze, freak

    if not os.path.exists(directory_out):
        os.system('mkdir ' + directory_out)

    master_img_name = image_list[0]
    master_img_dirs = master_img_name.split("/")
    img_master = cv2.imread(master_img_name)

    if master_0 == True:    # matching to master
        '''calculate descriptors in master image'''
        if descriptorVersion == "orb":
            keypoints_master, descriptor_master = OrbDescriptors(master_img_name, reduceKepoints)
            print('ORB descriptors calculated for master ' + master_img_dirs[-1])
        elif descriptorVersion == "sift":
            '''detect Harris keypoints in master image'''
            keypoints_master, _ = HarrisCorners(master_img_name, reduceKepoints, False)
            keypoints_master, descriptor_master = SiftDescriptors(master_img_name, keypoints_master)
            print('SIFT descriptors calculated for master ' + master_img_dirs[-1])
        elif descriptorVersion == "akaze":
            keypoints_master, descriptor_master = AKAZEDescriptors(master_img_name, reduceKepoints)
            print('AKAZE descriptors calculated for master ' + master_img_dirs[-1])


    '''perform co-registration for each image'''
    i = 1
    while i < len(image_list):

        slave_img_name = image_list[i]
        slave_img_dirs = slave_img_name.split("/")

        if master_0 is False:   # matching always to subsequent frame (no master)
            '''skip first image (because usage of subsequent images)'''
            if i == 1:
                i = i + 1
                continue

            if i == 2: # only once take both images from original folder
                '''calculate descriptors in master image'''
                if descriptorVersion == "orb":
                    keypoints_master, descriptor_master = OrbDescriptors(slave_img_name, reduceKepoints)
                    print('ORB descriptors calculated for master ' + slave_img_dirs[-1])
                elif descriptorVersion == "sift":
                    '''detect Harris keypoints in master image'''
                    keypoints_master, _ = HarrisCorners(slave_img_name, reduceKepoints, False)
                    keypoints_master, descriptor_master = SiftDescriptors(slave_img_name, keypoints_master)
                    print('SIFT descriptors calculated for master ' + slave_img_dirs[-1])
                elif descriptorVersion == "akaze":
                    keypoints_master, descriptor_master = AKAZEDescriptors(slave_img_name, reduceKepoints)
                    print('AKAZE descriptors calculated for master ' + slave_img_dirs[-1])
            else:  # co-registered image will be master for next image from original folder
                '''calculate descriptors in master image'''
                if descriptorVersion == "orb":
                    keypoints_master, descriptor_master = OrbDescriptors(img_coregistered, reduceKepoints, True)
                    print('ORB descriptors calculated for master ' + slave_img_dirs[-1])
                elif descriptorVersion == "sift":
                    '''detect Harris keypoints in master image'''
                    keypoints_master, _ = HarrisCorners(img_coregistered, reduceKepoints, True)
                    keypoints_master, descriptor_master = SiftDescriptors(img_coregistered, keypoints_master, True)
                    print('SIFT descriptors calculated for master ' + slave_img_dirs[-1])
                elif descriptorVersion == "akaze":
                    keypoints_master, descriptor_master = AKAZEDescriptors(img_coregistered, reduceKepoints, True)
                    print('AKAZE descriptors calculated for master ' + slave_img_dirs[-1])


        '''calculate ORB or SIFT descriptors in image to register'''
        if descriptorVersion == "orb":
            keypoints_image, descriptor_image = OrbDescriptors(slave_img_name, reduceKepoints)
            print('ORB descriptors calculated for image ' + slave_img_dirs[-1])
        elif descriptorVersion == "sift":
            '''detect Harris keypoints in image to register'''
            keypoints_image, _ = HarrisCorners(slave_img_name, reduceKepoints, False)
            keypoints_image, descriptor_image = SiftDescriptors(slave_img_name, keypoints_image)
            print('SIFT descriptors calculated for image ' + slave_img_dirs[-1])
        elif descriptorVersion == "akaze":
            keypoints_image, descriptor_image = AKAZEDescriptors(slave_img_name, reduceKepoints)
            print('AKAZE descriptors calculated for image ' + slave_img_dirs[-1])


        '''match images to master using feature descriptors'''
        if descriptorVersion == "orb":
            matched_pts_master, matched_pts_img = match_DescriptorsBF(descriptor_master, descriptor_image, keypoints_master, keypoints_image,
                                                                      True, feature_match_twosided)
            matched_pts_master = np.asarray(matched_pts_master, dtype=np.float32)
            matched_pts_img = np.asarray(matched_pts_img, dtype=np.float32)
        elif descriptorVersion == "sift":
            if feature_match_twosided:
                matched_pts_master, matched_pts_img = match_twosidedSift(descriptor_master, descriptor_image, keypoints_master, keypoints_image, "FLANN")
            else:
                matchscores = SiftMatchFLANN(descriptor_master, descriptor_image)
                matched_pts_master = np.float32([keypoints_master[m[0].queryIdx].pt for m in matchscores]).reshape(-1,2)
                matched_pts_img = np.float32([keypoints_image[m[0].trainIdx].pt for m in matchscores]).reshape(-1,2)
        elif descriptorVersion == "akaze":
            matched_pts_master, matched_pts_img = match_DescriptorsBF_NN(keypoints_master, keypoints_image, descriptor_master, descriptor_image, feature_match_twosided)

        print('number of matches: ' + str(matched_pts_master.shape[0]))


        '''calculate homography from matched image points and co-register images with estimated 3x3 transformation'''
        if matched_pts_master.shape[0] > nbr_good_matches:
            # Calculate Homography
            H_matrix, _ = cv2.findHomography(matched_pts_img, matched_pts_master, cv2.RANSAC, 3)

            # Warp source image to destination based on homography
            img_src = cv2.imread(slave_img_name)
            img_coregistered = cv2.warpPerspective(img_src, H_matrix, (img_master.shape[1],img_master.shape[0]))

            # save co-registered image
            cv2.imwrite(os.path.join(directory_out, slave_img_dirs[-1])[:-4] + '_coreg.jpg', img_coregistered)

            #drawF.drawPointsToImg(img_src, matched_pts_img, True)
            #plt.savefig(os.path.join(directory_out, slave_img_dirs[-1])[:-4] + '_features.png', dpi=600)

        i = i + 1


def coregistrationListOut(image_list, reduceKepoints=None, descriptorVersion="orb",
                          feature_match_twosided=False, nbr_good_matches=10):

    img_master = image_list[1]

    '''calculate ORB or SIFT descriptors in master image'''
    if descriptorVersion == 'orb':
        keypoints_master, descriptor_master = OrbDescriptors(img_master, reduceKepoints, True)
    elif descriptorVersion == 'sift':
        keypoints_master, _ = HarrisCorners(img_master, reduceKepoints, True)
        keypoints_master, descriptor_master = SiftDescriptors(img_master, keypoints_master, True)
    elif descriptorVersion == 'akaze':
        keypoints_master, descriptor_master = AKAZEDescriptors(img_master, reduceKepoints, True)
    print('descriptors calculated for master')

    '''perform co-registration for each image'''
    i = 1
    imgList = []
    while i < len(image_list):

        slave_img = image_list[i]

        '''calculate ORB or SIFT descriptors in image to register'''
        if descriptorVersion == 'orb':
            keypoints_image, descriptor_image = OrbDescriptors(slave_img, reduceKepoints, True)
        elif descriptorVersion == 'sift':
            keypoints_image, _ = HarrisCorners(slave_img, reduceKepoints, True)
            keypoints_image, descriptor_image = SiftDescriptors(slave_img, reduceKepoints, True)
        elif descriptorVersion == 'akaze':
            keypoints_image, descriptor_image = AKAZEDescriptors(slave_img, reduceKepoints, True)
        print('descriptors calculated for image ' + str(i))


        '''match images to master using feature descriptors (SIFT)'''
        if descriptorVersion == 'orb':
            matched_pts_master, matched_pts_img = match_DescriptorsBF(descriptor_master, descriptor_image, keypoints_master, keypoints_image,
                                                                      True, feature_match_twosided)
            matched_pts_master = np.asarray(matched_pts_master, dtype=np.float32)
            matched_pts_img = np.asarray(matched_pts_img, dtype=np.float32)
        elif descriptorVersion == "sift":
            if feature_match_twosided:
                matched_pts_master, matched_pts_img = match_twosidedSift(descriptor_master, descriptor_image, keypoints_master, keypoints_image, "FLANN")
            else:
                matchscores = SiftMatchFLANN(descriptor_master, descriptor_image)
                matched_pts_master = np.float32([keypoints_master[m[0].queryIdx].pt for m in matchscores]).reshape(-1,2)
                matched_pts_img = np.float32([keypoints_image[m[0].trainIdx].pt for m in matchscores]).reshape(-1,2)
        elif descriptorVersion == "akaze":
            matched_pts_master, matched_pts_img = match_DescriptorsBF_NN(keypoints_master, keypoints_image,
                                                                         descriptor_master, descriptor_image,
                                                                         feature_match_twosided)

        print('number of matches: ' + str(matched_pts_master.shape[0]))


        '''calculate homography from matched image points and co-register images with estimated 3x3 transformation'''
        if matched_pts_master.shape[0] > nbr_good_matches:
            # Calculate Homography
            H_matrix, _ = cv2.findHomography(matched_pts_img, matched_pts_master, cv2.RANSAC, 3)

            # Warp source image to destination based on homography
            img_coregistered = cv2.warpPerspective(slave_img, H_matrix, (img_master.shape[1],img_master.shape[0]))      #cv2.PerspectiveTransform() for points only

            imgList.append(img_coregistered)

        i = i + 1

    return imgList


# detect Harris corner features
def HarrisCorners(image, kp_nbr=None, img_import=False):

    if not img_import:
        image = cv2.imread(image)

    image_gray = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    '''detect Harris corners'''
    keypoints = cv2.cornerHarris(image_gray,2,3,0.04)
    keypoints = cv2.dilate(keypoints,None)

    # reduce keypoints to specific number
    thresh_kp_reduce = 0.01
    keypoints_prefilt = keypoints
    keypoints = np.argwhere(keypoints > thresh_kp_reduce * keypoints.max())

    if not kp_nbr == None:
        keypoints_reduced = keypoints
        while len(keypoints_reduced) >= kp_nbr:
            thresh_kp_reduce = thresh_kp_reduce + 0.01
            keypoints_reduced = np.argwhere(keypoints_prefilt > thresh_kp_reduce * keypoints_prefilt.max())
    else:
        keypoints_reduced = keypoints

    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints_reduced]

    return keypoints, keypoints_reduced #keypoints_reduced for drawing


# calculate ORB descriptors at detected features (using various feature detectors)
def OrbDescriptors(image, kp_nbr=None, img_import=False):
    if not img_import:
        image = cv2.imread(image)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    '''perform ORB'''
    if "4." in cv2.__version__ or "3." in cv2.__version__:
        if not kp_nbr == None:
            orb = cv2.ORB_create(nfeatures=kp_nbr)
        else:
            orb = cv2.ORB_create()
    else:
        orb = cv2.ORB()
    try:
        # find the keypoints with ORB
        keypoints = orb.detect(image_gray, None) #FAST detector
        keypoints, descriptors = orb.compute(image_gray, keypoints) #BRIEF descriptor
    except Exception as e:
        print(e)

    return keypoints, descriptors


# calculate SIFT descriptors at detected features (using various feature detectors)
def SiftDescriptors(image, keypoints, img_import=False):
    if not img_import:
        image = cv2.imread(image)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    '''perform SIFT'''
    if "3." in cv2.__version__ or "4." in cv2.__version__:
        siftCV2 = cv2.xfeatures2d.SIFT_create()
    else:
        siftCV2 = cv2.SIFT()
    keypoints, descriptors = siftCV2.compute(image_gray, keypoints)
    descriptors = descriptors.astype(np.uint8)

    return keypoints, descriptors

# SURF descirptors
def SurfDescriptors(image, HessThresh=400, img_import=False):
    if not img_import:
        image = cv2.imread(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    surf = cv2.xfeatures2d.SURF_create(HessThresh)
    surf.extended = True    # to calculate descriptor with dimension of 128 (instead of 64)

    # Find keypoints and descriptors directly
    keypoints, descriptors = surf.detectAndCompute(image_gray, None)

    return keypoints, descriptors

# FREAK descriptors
def FREAKDescriptors(image, keypoints, img_import=False):
    if not img_import:
        image = cv2.imread(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    freak = cv2.FREAK_create()
    keypoints, descriptors = freak.compute(image_gray, keypoints)
    descriptors = descriptors.astype(np.uint8)

    return keypoints, descriptors

# AKAZE descriptors
def AKAZEDescriptors(image, thresholdDetector=None, img_import=False):
    if not img_import:
        image = cv2.imread(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.uint8(image_gray)

    if not thresholdDetector == None:
        akaze = cv2.AKAZE_create(threshold=np.float(thresholdDetector))
    else:
        akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image_gray, None)

    return keypoints, descriptors

#match AKAZE
def match_DescriptorsBF_NN(kpts1, kpts2, desc1, desc2, twosided=True):
    # Use brute-force matcher to find 2-nn matches
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    # Use 2-nn matches and ratio criterion to find correct keypoint matches
    matched1 = []
    matched2 = []

    nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])

    if twosided:
        matched1_b = []
        matched2_b = []

        nn_matches_back = matcher.knnMatch(desc2, desc1, 2)
        for m, n in nn_matches_back:
            if m.distance < nn_match_ratio * n.distance:
                matched1_b.append(kpts2[m.queryIdx])
                matched2_b.append(kpts1[m.trainIdx])

        pts1_arr = np.asarray(matched1)
        pts2_arr = np.asarray(matched2)
        pts_12 = np.vstack((pts1_arr, pts2_arr)).T
        pts1_arr_b = np.asarray(matched1_b)
        pts2_arr_b = np.asarray(matched2_b)
        pts_21 = np.vstack((pts1_arr_b, pts2_arr_b)).T

        pts1_ts = []
        pts2_ts = []
        for pts in pts_12:
            pts_comp_1 = np.asarray(pts[0].pt, dtype=np.int)
            pts_comp_2 = np.asarray(pts[1].pt, dtype=np.int)
            for pts_b in pts_21:
                pts_comp_b_1 = np.asarray(pts_b[0].pt, dtype=np.int)
                pts_comp_b_2 = np.asarray(pts_b[1].pt, dtype=np.int)
                if ((pts_comp_1[0] == pts_comp_b_2[0]) and (pts_comp_1[1] == pts_comp_b_2[1])
                    and (pts_comp_2[0] == pts_comp_b_1[0]) and (pts_comp_2[1] == pts_comp_b_1[1])):
                    pts1_ts.append(pts[0].pt)
                    pts2_ts.append(pts[1].pt)

                    break

        pts1 = np.asarray(pts1_ts)
        pts2 = np.asarray(pts2_ts)

    return pts1, pts2

# match STAR image features using bruce force matching
# source code from Jan Erik Solem
def match_DescriptorsBF(des1,des2,kp1,kp2,ratio_test=True,twosided=True):
    '''Match STAR descriptors between two images'''
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    pts1 = []
    pts2 = []

    if ratio_test:
        # ratio test as per Lowe's paper
        good = []
        for m in matches:
            if m.distance < 100:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
    else:
        for m in matches:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if twosided:
        pts1_b = []
        pts2_b = []

        matches_back = bf.match(des2,des1)
        for m in matches_back:
            pts2_b.append(kp1[m.trainIdx].pt)
            pts1_b.append(kp2[m.queryIdx].pt)

        pts1_arr = np.asarray(pts1)
        pts2_arr = np.asarray(pts2)
        pts_12 = np.hstack((pts1_arr, pts2_arr))
        pts1_arr_b = np.asarray(pts1_b)
        pts2_arr_b = np.asarray(pts2_b)
        pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))


        pts1_ts = []
        pts2_ts = []
        for pts in pts_12:
            pts_comp = np.asarray(pts, dtype = np.int)
            for pts_b in pts_21:
                pts_b_comp = np.asarray(pts_b, dtype = np.int)
                if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                    and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                    pts1_ts.append(pts[0:2].tolist())
                    pts2_ts.append(pts[2:4].tolist())

                    break

        pts1 = pts1_ts
        pts2 = pts2_ts

        #print('Matches calculated')

    return pts1, pts2

# match SIFT image features using FLANN matching
# source code from Jan Erik Solem
def SiftMatchFLANN(des1, des2):
    max_dist = 0
    min_dist = 100

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)

    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    for m, n in matches:
        if min_dist > n.distance:
            min_dist = n.distance
        if max_dist < n.distance:
            max_dist = n.distance

    good = []
    for m, n in matches:
        if m.distance <= 3 * min_dist:
            good.append([m])

    return good


# match SIFT image features using bruce force matching
# source code from Jan Erik Solem
def SiftMatchBF(des1, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good

# match SIFT image features using FLANN matching and perform two-sided matching
# source code from Jan Erik Solem
def match_twosidedSift(desc1, desc2, kp1, kp2, match_Variant="FLANN"):
    '''Two-sided symmetric version of match().'''
    if match_Variant == "FLANN":
        matches_12 = SiftMatchFLANN(desc1, desc2)
        matches_21 = SiftMatchFLANN(desc2, desc1)
    elif match_Variant == "BF":
        matches_12 = SiftMatchBF(desc1, desc2)
        matches_21 = SiftMatchBF(desc2, desc1)

    pts1 = []
    pts2 = []
    for m in matches_12:
        pts1.append(kp1[m[0].queryIdx].pt)
        pts2.append(kp2[m[0].trainIdx].pt)

    pts1_b = []
    pts2_b = []
    for m in matches_21:
        pts2_b.append(kp1[m[0].trainIdx].pt)
        pts1_b.append(kp2[m[0].queryIdx].pt)

    pts1_arr = np.asarray(pts1)
    pts2_arr = np.asarray(pts2)
    pts_12 = np.hstack((pts1_arr, pts2_arr))
    pts1_arr_b = np.asarray(pts1_b)
    pts2_arr_b = np.asarray(pts2_b)
    pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))

    pts1_ts = []
    pts2_ts = []
    for pts in pts_12:
        pts_comp = np.asarray(pts, dtype=np.int)
        for pts_b in pts_21:
            pts_b_comp = np.asarray(pts_b, dtype=np.int)
            if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                    and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                pts1_ts.append(pts[0:2].tolist())
                pts2_ts.append(pts[2:4].tolist())
                break

    pts1 = np.asarray(pts1_ts, dtype=np.float32)
    pts2 = np.asarray(pts2_ts, dtype=np.float32)

    return pts1, pts2


def accuracy_coregistration(image_list_coreg, check_pts_img, output_dir, templateSize=30):
    first_image = True
    firstTrack = True
    maxDistBackForward_px = 1

    featuresToSearchFloat = np.asarray(check_pts_img[['x','y']], dtype=np.float32)
    # parameters for lucas kanade optical flow
    lk_paramsNoIntial = dict(winSize=(templateSize, templateSize), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    for image in image_list_coreg:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if first_image:
            imgMaster = img
            first_image = False
            continue

        trackedFeatures, status, _ = cv2.calcOpticalFlowPyrLK(imgMaster, img, featuresToSearchFloat,
                                                              None, **lk_paramsNoIntial)
        # check backwards
        trackedFeaturesCheck, status, _ = cv2.calcOpticalFlowPyrLK(img, imgMaster, trackedFeatures,
                                                                   None, **lk_paramsNoIntial)

        # set points that fail backward forward tracking test to nan
        distBetweenBackForward = abs(featuresToSearchFloat - trackedFeaturesCheck).reshape(-1, 2).max(-1)
        keepGoodTracks = distBetweenBackForward < maxDistBackForward_px
        #trackedFeaturesDF = pd.DataFrame(trackedFeatures, columns=['x', 'y'])
        trackedFeaturesDF = pd.DataFrame(check_pts_img.id)
        #trackedFeaturesDF.loc[:, 'id'] = check_pts_img.id.values
        trackedFeaturesDF.loc[:, 'dist'] = np.sqrt(np.square(featuresToSearchFloat[:,0]-trackedFeatures[:,0]) +
                                                   np.square(featuresToSearchFloat[:,1]-trackedFeatures[:,1]))
        trackedFeaturesDF.loc[:, 'check'] = keepGoodTracks
        trackedFeaturesDF = trackedFeaturesDF.where(trackedFeaturesDF.check == True)
        trackedFeaturesDF = trackedFeaturesDF.drop(['check'], axis=1)

        if firstTrack:
            trackedDistances = trackedFeaturesDF.copy()
            firstTrack = False
            continue

        trackedDistances = pd.concat([trackedDistances, trackedFeaturesDF])


    trackedDistances.to_csv(output_dir + 'trackedDistances.txt')

    stats = pd.DataFrame(check_pts_img.id)
    stats.loc[:, 'std'] = trackedDistances.groupby('id').dist.std()
    stats.loc[:, 'mean'] = trackedDistances.groupby('id').dist.mean()
    stats.loc[:, 'min'] = trackedDistances.groupby('id').dist.min()
    stats.loc[:, 'max'] = trackedDistances.groupby('id').dist.max()
    stats.loc[:, 'median'] = trackedDistances.groupby('id').dist.median()

    stats.to_csv(output_dir + 'statsCoregAcc.txt')

    '''draw check point locations'''
    drawF.draw_points_onto_image(imgMaster, np.asarray(check_pts_img[['x','y']]), np.asarray(check_pts_img.id), 5, 15)
    plt.savefig(os.path.join(output_dir, 'accuracy_coreg_checkPts_location.jpg'), dpi=600)


# define template at image point position (of corresponding GCP)
def getTemplateAtImgpoint(img, img_pts, template_width=10, template_height=10):
# consideration that row is y and column is x
# careful that template extent even to symmetric size around point of interest

    template_img = []
    anchor_pts = []
    for pt in img_pts:
        if img_pts.shape[1] > 2:
            template_width_for_cut_left = pt[2]/2
            template_width_for_cut_right = pt[2]/2 + 1
        elif template_width > 0:
            template_width_for_cut_left = template_width/2
            template_width_for_cut_right = template_width/2 + 1
        else:
            print('missing template size assignment')

        if img_pts.shape[1] > 2:
            template_height_for_cut_lower = pt[3]/2
            template_height_for_cut_upper = pt[3]/2 + 1
        elif template_height > 0:
            template_height_for_cut_lower = template_height/2
            template_height_for_cut_upper = template_height/2 + 1
        else:
            print('missing template size assignment')

        cut_anchor_x = pt[0] - template_width_for_cut_left
        cut_anchor_y = pt[1] - template_height_for_cut_lower

        #consideration of reaching of image boarders (cutting of templates)
        if pt[1] + template_height_for_cut_upper > img.shape[0]:
            template_height_for_cut_upper = np.int(img.shape[0] - pt[1])
        if pt[1] - template_height_for_cut_lower < 0:
            template_height_for_cut_lower = np.int(pt[1])
            cut_anchor_y = 0
        if pt[0] + template_width_for_cut_right > img.shape[1]:
            template_width_for_cut_right = np.int(img.shape[1] - pt[0])
        if pt[0] - template_width_for_cut_left < 0:
            template_width_for_cut_left = np.int(pt[0])
            cut_anchor_x = 0

        template = img[np.int(pt[1]-template_height_for_cut_lower) : np.int(pt[1]+template_height_for_cut_upper),
                       np.int(pt[0]-template_width_for_cut_left) : np.int(pt[0]+template_width_for_cut_right)]

        #template_img = np.dstack((template_img, template))
        template_img.append(template)

        anchor_pts.append([cut_anchor_x, cut_anchor_y])

    anchor_pts = np.asarray(anchor_pts, dtype=np.float32)
    #template_img = np.delete(template_img, 0, axis=2)

    return template_img, anchor_pts #anchor_pts defines position of lower left of template in image


# template matching for automatic detection of image coordinates of GCPs
def performTemplateMatch(img_extracts, template_img, anchor_pts, plot_results=False):
    new_img_pts = []
    template_nbr = 0

    count_pts = 0
    while template_nbr < len(template_img):
        template_array = np.asarray(template_img[template_nbr])
        if (type(img_extracts) is list and len(img_extracts) > 1) or (type(img_extracts) is tuple and len(img_extracts.shape) > 2):
            img_extract = img_extracts[template_nbr]
        else:
            img_extract = img_extracts
        res = cv2.matchTemplate(img_extract, template_array, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #min_loc for TM_SQDIFF
        match_position_x = max_loc[0] + template_array.shape[1]/2
        match_position_y = max_loc[1] + template_array.shape[0]/2
        del min_val, min_loc

        if max_val > 0.9:
            new_img_pts.append([match_position_x + anchor_pts[template_nbr,0],
                                match_position_y + anchor_pts[template_nbr,1]])
            count_pts = count_pts + 1

        template_nbr = template_nbr + 1

        if plot_results:
            plt.subplot(131),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.plot(match_position_x-template_array.shape[1]/2, match_position_y-template_array.shape[0]/2, "r.", markersize=10)
            plt.subplot(132),plt.imshow(img_extract,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.plot(match_position_x, match_position_y, "r.", markersize=10)
            plt.subplot(133),plt.imshow(template_array,cmap = 'gray')
            plt.title('Template'), plt.xticks([]), plt.yticks([])
            plt.show()

    new_img_pts = np.asarray(new_img_pts, dtype=np.float32)
    new_img_pts = new_img_pts.reshape(count_pts, 2)

    return new_img_pts
