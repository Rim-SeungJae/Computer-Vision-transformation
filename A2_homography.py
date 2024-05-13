import cv2
import numpy as np
import math
import time
import random

def hamming_distance(a, b):
    return sum(i != j for i, j in zip(a, b))

def match_des(des1, des2):
    matches = []
    for i, des1_i in enumerate(des1):
        min = len(des1_i) + 1
        sec_min = min
        queryIdx = -1
        trainIdx = -1
        for j, des2_j in enumerate(des2):
            dist = hamming_distance(des1_i, des2_j)
            if dist < min:
                sec_min = min
                min = dist
                queryIdx = i
                trainIdx = j

        cross_min = len(des1_i) + 1
        cross_queryIdx = -1
        for j, des1_j in enumerate(des1):
            dist = hamming_distance(des1_j, des2[trainIdx])
            if dist < cross_min:
                cross_min = dist
                cross_queryIdx = j

        dmatch = cv2.DMatch(queryIdx, trainIdx, float(min))
        # cross validate and threshold
        if cross_queryIdx == queryIdx and min / sec_min <= 0.9:
            matches.append(dmatch)
    return matches

def compute_homography(srcP, destP):
    N, _ = srcP.shape

    srcP_init = srcP.copy()
    destP_init = destP.copy()

    srcP_sum = srcP.sum(axis=0)
    x_mean, y_mean = srcP_sum[0]/N, srcP_sum[1]/N
    max_dist = -1

    for i in srcP:
        i[0] -= x_mean
        i[1] -= y_mean
        if math.sqrt(i[0]**2 + i[1]**2) > max_dist:
            max_dist = math.sqrt(i[0]**2 + i[1]**2)

    for i in srcP:
        i[0] /= max_dist/math.sqrt(2)
        i[1] /= max_dist/math.sqrt(2)

    TS = [
        [math.sqrt(2) / max_dist, 0, -x_mean * math.sqrt(2) / max_dist],
        [0, math.sqrt(2) / max_dist, -y_mean * math.sqrt(2) / max_dist],
        [0, 0, 1]
    ]
    TS = np.array(TS)

    destP_sum = destP.sum(axis=0)
    x_mean, y_mean = destP_sum[0] / N, destP_sum[1] / N
    max_dist = -1

    for i in destP:
        i[0] -= x_mean
        i[1] -= y_mean
        if math.sqrt(i[0]**2 + i[1]**2) > max_dist:
            max_dist = math.sqrt(i[0]**2 + i[1]**2)

    for i in destP:
        i[0] /= max_dist/math.sqrt(2)
        i[1] /= max_dist/math.sqrt(2)

    TD = [
        [math.sqrt(2) / max_dist, 0, -x_mean * math.sqrt(2) / max_dist],
        [0, math.sqrt(2) / max_dist, -y_mean * math.sqrt(2) / max_dist],
        [0, 0, 1]
    ]
    TD = np.array(TD)

    A = []

    for i in range(N):
        src = srcP[i]
        dest = destP[i]

        Ai = [
            [-src[0], -src[1], -1, 0, 0, 0, src[0] * dest[0], src[1] * dest[0], dest[0]],
            [0, 0, 0, -src[0], -src[1], -1, src[0] * dest[1], src[1] * dest[1], dest[1]]
        ]

        A += Ai

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)

    min_idx = np.argmin(s)
    H = vh[min_idx].reshape((3,3))

    H_denorm = np.dot(np.dot(np.linalg.inv(TD), H), TS)
    H_denorm /= H_denorm[2][2]

    # this is for testing
    #for i in range(N):
    #    test = [srcP_init[i][0], srcP_init[i][1], 1]
    #    result = np.dot(H_denorm, test)
    #    print(result/result[2])
    #    print(destP_init[i])

    return H_denorm

def compute_homography_ransac(srcP, destP, th):
    time_i = time.time()
    H_max = []
    inliers_max = -1
    inliers_max_list = []
    while True:
        indices = random.sample(range(len(srcP)), 4)

        srcP_sampled = srcP[np.array(indices)]
        destP_sampled = destP[np.array(indices)]

        H_cur = compute_homography(srcP_sampled, destP_sampled)

        inliers_cnt = 0
        inliers_list = []
        for i, src in enumerate(srcP):
            dest = destP[i]
            homo_location = [src[0], src[1], 1]

            warped_location = np.dot(H_cur, homo_location)
            warped_location /= warped_location[2]
            warped_location = warped_location[:2]

            dx = dest[0] - warped_location[0]
            dy = dest[1] - warped_location[1]
            dist = math.sqrt(dx**2+dy**2)
            if dist <= th:
                inliers_cnt += 1
                inliers_list.append(i)

        if inliers_cnt > inliers_max:
            H_max = H_cur
            inliers_max_list = inliers_list
            inliers_max = inliers_cnt

        if time.time() - time_i > 2.7:
            break

    srcP_inliers = srcP[np.array(inliers_max_list)]
    destP_inliers = destP[np.array(inliers_max_list)]

    H_result = compute_homography(srcP_inliers, destP_inliers)

    return H_result


if __name__=="__main__":
    img1 = cv2.imread('./cv_desk.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    img_hp = cv2.imread('./hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    img_d10 = cv2.imread('./diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
    img_d11 = cv2.imread('./diamondhead-11.png', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1 = orb.detect(img1, None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2 = orb.detect(img2, None)
    kp2, des2 = orb.compute(img2, kp2)

    matches = match_des(des1, des2)

    matches= sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    cv2.imshow('top-10 matches',img3)
    cv2.waitKey()

    # Prepare matches for homography with normalization
    # Remove incorrect matches
    matches_for_norm = np.delete(matches, [10,12])

    srcP = []
    destP = []

    for match in matches_for_norm[:15]:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = kp1[idx1].pt
        x2, y2 = kp2[idx2].pt

        destP.append([x1, y1])
        srcP.append([x2, y2])

    srcP = np.array(srcP, dtype=np.float32)
    destP = np.array(destP, dtype=np.float32)

    H = compute_homography(srcP, destP)

    warpped = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    composed = img1.copy()
    for i,row in enumerate(warpped):
        for j, element in enumerate(row):
            if element != 0:
                composed[i][j] = element

    cv2.imshow('Homography with normalization-warped', warpped)
    cv2.imshow('Homography with normalization-composed', composed)
    cv2.waitKey()

    srcP = []
    destP = []

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = kp1[idx1].pt
        x2, y2 = kp2[idx2].pt

        destP.append([x1, y1])
        srcP.append([x2, y2])

    srcP = np.array(srcP, dtype=np.float32)
    destP = np.array(destP, dtype=np.float32)

    H = compute_homography_ransac(srcP, destP, 10)

    warpped = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    composed = img1.copy()
    for i, row in enumerate(warpped):
        for j, element in enumerate(row):
            if element != 0:
                composed[i][j] = element

    cv2.imshow('Homography with RANSAC-warped', warpped)
    cv2.imshow('Homography with RANSAC-composed', composed)
    cv2.waitKey()

    # resize harry poter cover image
    img_hp = cv2.resize(img_hp, dsize=(img2.shape[1], img2.shape[0]))

    warpped = cv2.warpPerspective(img_hp, H, (img1.shape[1], img1.shape[0]))
    composed = img1.copy()
    for i, row in enumerate(warpped):
        for j, element in enumerate(row):
            if element != 0:
                composed[i][j] = element

    cv2.imshow('harry potter homography-warped', warpped)
    cv2.imshow('harry potter homography-composed', composed)
    cv2.waitKey()

    ########################################################################
    # from here, image stitching
    ########################################################################

    empty = np.zeros(img_d10.shape, dtype=np.uint8)
    img_d11 = cv2.hconcat([empty, img_d11])

    orb = cv2.ORB_create()
    kp1 = orb.detect(img_d11, None)
    kp1, des1 = orb.compute(img_d11, kp1)
    kp2 = orb.detect(img_d10, None)
    kp2, des2 = orb.compute(img_d10, kp2)

    matches = match_des(des1, des2)

    matches= sorted(matches, key=lambda x: x.distance)

    srcP = []
    destP = []

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = kp1[idx1].pt
        x2, y2 = kp2[idx2].pt

        destP.append([x1, y1])
        srcP.append([x2, y2])

    srcP = np.array(srcP, dtype=np.float32)
    destP = np.array(destP, dtype=np.float32)

    H = compute_homography_ransac(srcP, destP, 10)

    warpped = cv2.warpPerspective(img_d10, H, (img_d11.shape[1], img_d11.shape[0]))
    composed = img_d11.copy()

    boundary_leftmost = composed.shape[1] + 1
    boundary_rightmost = -1
    overlap_leftmost = composed.shape[1] + 1
    overlap_rightmost = -1
    for i, row in enumerate(warpped):
        for j, element in enumerate(row):
            if element != 0 and composed[i][j] != 0:
                if j < overlap_leftmost:
                    overlap_leftmost = j
                if j > overlap_rightmost:
                    overlap_rightmost = j

    gap = overlap_rightmost-overlap_leftmost

    # gradation based blending
    for i, row in enumerate(warpped):
        for j, element in enumerate(row):
            if element != 0:
                if composed[i][j] != 0:
                    alpha = (j-overlap_leftmost)/gap
                    composed[i][j] = alpha * element + (1 - alpha) * composed[i][j]
                else:
                    composed[i][j] = element

    black_idx = 0
    for i in range(composed.shape[1]):
        if composed[:, i].sum() != 0:
            black_idx = i
            break

    composed = composed[:, black_idx:]

    cv2.imshow('diamondhead-composed, blending applied', composed)
    cv2.waitKey()
