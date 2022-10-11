import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic,mark_boundaries
from sklearn import preprocessing
from numba import jit

def LDA(data, gt, height, width, bands):

    data_2d = np.reshape(data, [height * width, bands])
    gt = np.reshape(gt, [-1])
    index = np.where(gt != 0)[0]
    data_samples = data_2d[index]
    gt_samples = gt[index]
    lda = LinearDiscriminantAnalysis()
    lda.fit(data_samples, gt_samples - 1)
    LDA_result = lda.transform(data_2d)

    return np.reshape(LDA_result, [height, width, -1])

def SLIC_seg(LDA_result, gt, height, width, bands, superpixel_num):

    print("superpixel_num", superpixel_num)
    segments = slic(LDA_result, n_segments=superpixel_num, compactness=5, max_iter=20,
                    convert2lab=False, sigma=0, enforce_connectivity=True,
                    min_size_factor=0.3, max_size_factor=2, slic_zero=False)
    superpixel_count = segments.max() + 1
    print("superpixel_count", superpixel_count)
    out = mark_boundaries(LDA_result[:, :, 0], segments)
    plt.figure()
    plt.imshow(out)
    plt.show()
    segments_1d = np.reshape(segments, [-1])
    S = np.zeros([superpixel_count, 1], dtype=np.float32)
    Q = np.zeros([height * width, superpixel_count], dtype=np.float32)
    x = np.reshape(LDA_result, [-1, 1])

    for i in range(superpixel_count):
        idx = np.where(segments_1d == i)[0]
        count = len(idx)
        pixels = x[idx]
        superpixel = np.sum(pixels, 0) / count
        S[i] = superpixel
        Q[idx, i] = 1

    A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
    for i in range(height - 2):
        for j in range(width - 2):
            sub = segments[i:i + 2, j:j + 2]
            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue
                pix1 = S[idx1]
                pix2 = S[idx2]
                diss = np.exp(-np.sum(np.square(
                    pix1 - pix2)) / 10 ** 2)
                A[idx1, idx2] = A[idx2, idx1] = diss
    A = preprocessing.scale(A, axis=1)
    A = A + np.eye(superpixel_count)
    return Q, S, A, segments
@jit
def get_temporal_adj(pixel_before, pixel_after):
    count_before = np.size(pixel_before)
    count_after = np.size(pixel_after)
    adj = np.zeros([count_before, count_after])
    for i in range(count_before):
        pix1 = pixel_before[i]
        for j in range(count_after):
            if adj[i,j] != 0 :
                continue
            pix2 = pixel_after[j]
            diss = np.exp(-np.sum(np.square(pix1 - pix2)) / 10 ** 2)
            adj[i,j] = diss
    adj1 = preprocessing.scale(adj,axis=1)
    adj2 = preprocessing.scale(adj.T,axis=1)
    return adj1, adj2
def seg_module(data, gt, superpixel_num):

    height, width, bands = data.shape
    LDA_result = LDA(data, gt, height, width, bands)
    Q, S, A, seg = SLIC_seg(LDA_result, gt, height, width, bands, superpixel_num )
    supernum = Q.shape[1]
    return Q, S, A, seg, supernum



