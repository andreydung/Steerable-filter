# import Steerable.Steerable as Steerable
from __future__ import division
import numpy as np
import Steerable as Steerable
import cv2

def mse(img1, img2):
	return ((img2 - img1)**2).mean()

def fspecial(win = 11, sigma = 1.5):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	shape = (win, win)
	m, n = [(ss-1.)/2. for ss in shape]
	y, x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()

	if sumh != 0:
		h /= sumh
	return h

def STSIM_M(im1, im2):
	pass

def STSIM2(im1, im2):
	pass

class Metric:

	def __init__(self):
		self.win = 7

	def ssim(self, img1, img2, K = (0.01, 0.03), L = 255):

		img1 = img1.astype(float)
		img2 = img2.astype(float)

		C1 = (K[0]*L) ** 2
		C2 = (K[1]*L) ** 2
		
		window = fspecial()
		window /= window.sum()

		mu1 = self.conv( img1, window)
		mu2 = self.conv( img2, window)

		mu1_sq = mu1 * mu1
		mu2_sq = mu2 * mu2
		mu1_mu2 = mu1 * mu2
		
		sigma1_sq = self.conv( img1*img1, window) - mu1_sq;
		sigma2_sq = self.conv( img2*img2, window) - mu2_sq;
		sigma12 = self.conv( img1*img2, window) - mu1_mu2;

		ssim_map = 	((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

		return ssim_map.mean()

	def conv(self, a, b):
		"""
		Larger matrix go first
		"""
		return cv2.filter2D(a, -1, b, anchor = (0,0))\
			[:(a.shape[0]-b.shape[0]+1), :(a.shape[1]-b.shape[1]+1)]

	def STSIM(self, im1, im2):
		s = Steerable.Steerable()

		pyrA = s.buildSFpyr(im1)
		pyrB = s.buildSFpyr(im2)

		pyrA = s.getlist(pyrA)
		pyrB = s.getlist(pyrB)

		stsim = map(self.pooling, pyrA, pyrB)

		return np.mean(stsim)

	def pooling(self, im1, im2):
		win = self.win
		tmp = self.compute_L_term(im1, im2) * self.compute_C_term(im1, im2) * \
			self.compute_C01_term(im1, im2) * self.compute_C10_term(im1, im2)

		return np.power(tmp.mean(), 0.25)

	def compute_L_term(self, im1, im2):
		assert ((im1.dtype == float) and (im2.dtype == float))

		win = self.win
		C = 0.001
		window = fspecial(win, win/6)
		mu1 = np.abs(self.conv(im1, window))
		mu2 = np.abs(self.conv(im2, window))

		Lmap = (2 * mu1 * mu2 + C)/(mu1*mu1 + mu2*mu2 + C)
		return Lmap

	def compute_C_term(self, im1, im2):
		assert ((im1.dtype == float) and (im2.dtype == float))
		
		win = self.win
		C = 0.001
		window = fspecial(win, win/6)
		mu1 = np.abs(self.conv(im1, window))
		mu2 = np.abs(self.conv(im2, window))

		sigma1_sq = self.conv(im1*im1, window) - mu1 * mu1
		sigma1 = np.sqrt(sigma1_sq)
		sigma2_sq = self.conv(im2*im2, window) - mu2 * mu2
		sigma2 = np.sqrt(sigma2_sq)

		Cmap = (2*sigma1*sigma2 + C)/(sigma1_sq + sigma2_sq + C)
		return Cmap

	def compute_C01_term(self, im1, im2):
		assert ((im1.dtype == float) and (im2.dtype == float))
		win = self.win
		C = 0.001;
		window2 = 1/(win*(win-1)) * np.ones((win,win-1));

		im11 = im1[:, :-1]
		im12 = im1[:, 1:]
		im21 = im2[:, :-1]
		im22 = im2[:, 1:]

		mu11 = self.conv(im11, window2)
		mu12 = self.conv(im12, window2)
		mu21 = self.conv(im21, window2)
		mu22 = self.conv(im22, window2)

		sigma11_sq = self.conv(im11*im11, window2) - mu11*mu11
		sigma12_sq = self.conv(im12*im12, window2) - mu12*mu12
		sigma21_sq = self.conv(im21*im21, window2) - mu21*mu21
		sigma22_sq = self.conv(im22*im22, window2) - mu22*mu22

		sigma1_cross = self.conv(im11*np.conj(im12), window2) - mu11*np.conj(mu12)
		sigma2_cross = self.conv(im21*np.conj(im22), window2) - mu21*np.conj(mu22)

		rho1 = (sigma1_cross + C)/(np.sqrt(sigma11_sq)*np.sqrt(sigma12_sq) + C)
		rho2 = (sigma2_cross + C)/(np.sqrt(sigma21_sq)*np.sqrt(sigma22_sq) + C)
		C01map = 1 - 0.5*np.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, im1, im2):
		assert ((im1.dtype == float) and (im2.dtype == float))
		win = self.win
		C = 0.001;
		window2 = 1/(win*(win-1)) * np.ones((win-1,win));

		im11 = im1[:-1, :]
		im12 = im1[1:, :]
		im21 = im2[:-1, :]
		im22 = im2[1:, :]

		mu11 = self.conv(im11, window2)
		mu12 = self.conv(im12, window2)
		mu21 = self.conv(im21, window2)
		mu22 = self.conv(im22, window2)

		sigma11_sq = self.conv(im11*im11, window2) - mu11*mu11
		sigma12_sq = self.conv(im12*im12, window2) - mu12*mu12
		sigma21_sq = self.conv(im21*im21, window2) - mu21*mu21
		sigma22_sq = self.conv(im22*im22, window2) - mu22*mu22

		sigma1_cross = self.conv(im11*np.conj(im12), window2) - mu11*np.conj(mu12)
		sigma2_cross = self.conv(im21*np.conj(im22), window2) - mu21*np.conj(mu22)

		rho1 = (sigma1_cross + C)/(np.sqrt(sigma11_sq)*np.sqrt(sigma12_sq) + C)
		rho2 = (sigma2_cross + C)/(np.sqrt(sigma21_sq)*np.sqrt(sigma22_sq) + C)
		C10map = 1 - 0.5*np.abs(rho1 - rho2)

		return C10map

	# def compute_cross_term(self, im11, im12, im21, im22):

	# 	C = 0.001;
	# 	window2 = 1/(win**2)*np.ones((win, win));

	# 	mu11 = (filter2(window2,im11,'valid'));
	# 	mu12 = (filter2(window2,im12,'valid'));

	# 	mu21 = (filter2(window2,im21,'valid'));
	# 	mu22 = (filter2(window2,im22,'valid'));


	# 	sigma11_sq = filter2(window2, (im11*im11), 'valid') - (mu11.*mu11);
	# 	sigma12_sq = filter2(window2, (im12.*im12), 'valid') - (mu12.*mu12);
	# 	sigma21_sq = filter2(window2, (im21.*im21), 'valid') - (mu21.*mu21);
	# 	sigma22_sq = filter2(window2, (im22.*im22), 'valid') - (mu22.*mu22);
	# 	sigma1_cross = filter2(window2,im11.*(im12),'valid') - mu11.*(mu12);
	# 	sigma2_cross = filter2(window2,im21.*(im22),'valid') - mu21.*(mu22);

	# 	rho1 = (sigma1_cross + C)./(sqrt(sigma11_sq).*sqrt(sigma12_sq) + C);
	# 	rho2 = (sigma2_cross + C)./(sqrt(sigma21_sq).*sqrt(sigma22_sq) + C);

	# 	Crossmap = 1 - 0.5*abs(rho1 - rho2);
	# 	return Crossmap