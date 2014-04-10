import cv2
import numpy as np
from metric import Metric
from Steerable import *
# from mlabwrap import mlab

ss = Metric()

# # TEST SSIM
# im1 = cv2.imread('images/10.tif', cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('images/02.tif', cv2.IMREAD_GRAYSCALE)

# result1 = ss.ssim(im1, im2)
# result2 = mlab.ssim(im1, im2)
# np.testing.assert_almost_equal(result1, result2, decimal = 10)

# TEST helper functions
# im1 = cv2.imread('images/10.tif', cv2.IMREAD_GRAYSCALE).astype(float)
# im2 = cv2.imread('images/02.tif', cv2.IMREAD_GRAYSCALE).astype(float)

# win = 7

# result1 = ss.compute_C01_term(im1, im2)
# print result1

# result2 = mlab.compute_C01_term(im1, im2, win)
# print result2
# np.testing.assert_almost_equal(result1, result2, decimal = 5)

# TEST STEERABLE
# s = Steerable()

# result1 = s.rcosFn(1, -0.5)
# result2 = mlab.rcosFn(1, -0.5)

# np.testing.assert_almost_equal(result1, result2, decimal = 5)


# TEST STSIM 
# im1 = cv2.imread('images/01.tif', cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('images/02.tif', cv2.IMREAD_GRAYSCALE)

# result1 = ss.STSIM(im1, im2)
# result2 = mlab.stsim2(im1, im2)

# print result1
# print result2

# TEST no sub pyramid
# im = cv2.imread('images/01.tif', cv2.IMREAD_GRAYSCALE)
# s_nosub = Steerable_noSub()

# coeff = s_nosub.buildSFpyr(im)

# print coeff[0].shape
# print coeff[1][0].shape
# print coeff[2][0].shape
# print coeff[3].shape

# TEST STSIM 
im1 = cv2.imread('images/01.tif', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('images/02.tif', cv2.IMREAD_GRAYSCALE)

result1 = ss.STSIM2(im1, im2)
print result1

print ss.STSIM_Maha(im1, im1)