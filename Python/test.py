import unittest
import numpy as np
from Steerable import *
from metric import Metric
import cv2
from mlabwrap import mlab

class TestSteerableNoSub(unittest.TestCase):

	def testNoSub(self):
		Nsc = 3
		s = SteerableNoSub(Nsc + 2)

		# Test lowpass and highpass
		im = cv2.imread('images/01.tif', cv2.IMREAD_GRAYSCALE)
		coeff1 = s.buildSCFpyr(im)

		# Highpass
		coeff2 = mlab.testNoSubbands(im, 0, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[0], decimal = 5)

		# Lowpass
		coeff2 = mlab.testNoSubbands(im, -1, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[-1], decimal = 5)

		# Bandpass
		coeff2 = mlab.testNoSubbands(im, 1, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[1][0], decimal = 5)
		coeff2 = mlab.testNoSubbands(im, 2, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[1][1], decimal = 5)

class TestSteerable(unittest.TestCase):

	def testDifferentSize(self):
		im = np.random.randint(0, 255, (128, 64))
		a = Steerable()
		coeff = a.buildSCFpyr(im)
		out = a.reconSCFpyr(coeff)

		self.assertEqual(np.allclose(out, im, atol = 10), True)

	def testutility(self):
		im = cv2.imread('images/05.tif', cv2.IMREAD_GRAYSCALE)
		a = Steerable()
		coeff = a.buildSCFpyr(im)

		cv2.imwrite('subbands.png', visualize(coeff))

	def testHelpers(self):
		s = Steerable()
		result1 = s.rcosFn(1, -0.5)[0]
		result2 = mlab.rcosFn(1, -0.5)[0]
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

		log_rad, angle = s.base(65,128)
		angle2 = mlab.testbase(65,128)

		np.testing.assert_almost_equal(angle, angle2, decimal = 5)

	def testSubbands(self):
		Nsc = 3
		s = Steerable(Nsc + 2)

		# Test lowpass and highpass
		im = cv2.imread('images/11.tif', cv2.IMREAD_GRAYSCALE)
		coeff1 = s.buildSCFpyr(im)

		# Highpass
		coeff2 = mlab.testsubbands(im, 0, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[0], decimal = 5)

		# Lowpass
		coeff2 = mlab.testsubbands(im, -1, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[-1], decimal = 5)

		# Bandpass
		coeff2 = mlab.testsubbands(im, 1, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[1][0], decimal = 5)
		coeff2 = mlab.testsubbands(im, 2, Nsc, 4)
		np.testing.assert_almost_equal(coeff2, coeff1[1][1], decimal = 5)

class TestMetric(unittest.TestCase):

	def testMetric(self):
		ss = Metric()
		im1 = cv2.imread('images/12.tif', cv2.IMREAD_GRAYSCALE).astype(float)
		im2 = cv2.imread('images/02.tif', cv2.IMREAD_GRAYSCALE).astype(float)

		a1 = ss.STSIM(im1, im2)
		a2 = mlab.stsim(im1, im2)
		np.testing.assert_almost_equal(a1,a2, decimal = 5)

		b1 = ss.STSIM2(im1, im2)
		b2 = mlab.stsim2(im1, im2)
		np.testing.assert_almost_equal(b1,b2, decimal = 5)

	def testHelpers(self):

		ss = Metric()
		im1 = cv2.imread('images/08.tif', cv2.IMREAD_GRAYSCALE).astype(float)
		im2 = cv2.imread('images/10.tif', cv2.IMREAD_GRAYSCALE).astype(float)
		im3 = cv2.imread('images/01.tif', cv2.IMREAD_GRAYSCALE).astype(float)
		im4 = cv2.imread('images/07.tif', cv2.IMREAD_GRAYSCALE).astype(float)

		win = 7

		result1 = ss.compute_L_term(im1, im2)
		result2 = mlab.compute_L_term(im1, im2, win)
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

		result1 = ss.compute_C_term(im1, im2)
		result2 = mlab.compute_C_term(im1, im2, win)
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

		result1 = ss.compute_C01_term(im1, im2)
		result2 = mlab.compute_C01_term(im1, im2, win)
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

		result1 = ss.compute_C10_term(im1, im2)
		result2 = mlab.compute_C10_term(im1, im2, win)
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

		result1 = ss.compute_cross_term(im1, im2, im3, im4)
		result2 = mlab.compute_Cross_term(im1, im2, im3, im4, win)
		np.testing.assert_almost_equal(result1, result2, decimal = 5)

if __name__ == "__main__":
	unittest.main()
