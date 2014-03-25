# import Steerable.Steerable as Steerable
import numpy as np
import scipy.signal as sc


def mse(img1, img2):
	return ((img2 - img1)**2).mean()

def ssim(img1, img2, K = (0.01, 0.03), L = 255):

	img1 = img1.astype(float)
	img2 = img2.astype(float)

	C1 = (K[0]*L) ** 2
	C2 = (K[1]*L) ** 2
	
	window = fspecial()
	window /= window.sum()

	mu1   = sc.convolve2d( img1, window, mode ='valid')
	mu2   = sc.convolve2d( img2, window, mode = 'valid')

	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	
	sigma1_sq = sc.convolve2d( img1*img1, window,  mode = 'valid') - mu1_sq;
	sigma2_sq = sc.convolve2d( img2*img2, window,  mode = 'valid') - mu2_sq;
	sigma12 = sc.convolve2d( img1*img2, window, mode = 'valid') - mu1_mu2;

	ssim_map = 	((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	return ssim_map.mean()

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

def compute_L_term(im1, im2, win):
	assert ((im1.dtype == float) and (im2.dtype == float))

	C = 0.001
	window = fspecial(win, win/6)
	mu1 = np.abs(sc.convolve2d(im1, window, mode = 'valid'))
	mu2 = np.abs(sc.convolve2d(im2, window, mode = 'valid'))

	Lmap = (2 * mu1 * mu2 + C)/(mu1*mu1 + mu2*mu2 + C)
	return Lmap

def compute_C_term(im1, im2, win):
	assert ((im1.dtype == float) and (im2.dtype == float))

	C = 0.001
	window = fspecial(win, win/6)
	mu1 = np.abs(sc.convolve2d(im1, window, mode = 'valid'))
	mu2 = np.abs(sc.convolve2d(im2, window, mode = 'valid'))

	sigma1_sq = sc.convolve2d(im1*im1, window, mode = 'valid') - mu1 * mu1
	sigma1 = np.sqrt(sigma1_sq)
	sigma2_sq = sc.convolve2d(im2*im2, window, mode = 'valid') - mu2 * mu2
	sigma2 = np.sqrt(sigma2_sq)

	Cmap = (2*sigma1*sigma2 + C)/(sigma1_sq + sigma2_sq + C)
	return Cmap

def compute_C01_term(im1, im2, win):
	assert ((im1.dtype == float) and (im2.dtype == float))
	C = 0.001;
	window2 = 1/(win*(win-1)) * np.ones(win,win-1);

	im11 = im1[:, : end-1]
	im12 = im1[:, 1:]
	im21 = im2[:, :end-1]
	im22 = im2[:, 1:]

	mu11 = sc.convolve2d(im11, window2, mode = 'valid')
	mu12 = sc.convolve2d(im12, window2, mode = 'valid')
	mu21 = sc.convolve2d(im21, window2, mode = 'valid')
	mu22 = sc.convolve2d(im22, window2, mode = 'valid')

	sigma11_sq = sc.convolve2d(im11*im11, window2, mode = 'valid') - mu11*mu11
	sigma11_sq = sc.convolve2d(im12*im12, window2, mode = 'valid') - mu12*mu12
	sigma11_sq = sc.convolve2d(im21*im21, window2, mode = 'valid') - mu21*mu21
	sigma11_sq = sc.convolve2d(im22*im22, window2, mode = 'valid') - mu22*mu22

	sigma1_cross = sc.convolve2d(im11*np.conj(im12), window2, mode = 'valid') - mu11*np.conj(mu12)
	sigma2_cross = sc.convolve2d(im21*np.conj(im22), window2, mode = 'valid') - mu21*np.conj(mu12)

	rho1 = (sigma1_cross + C)./(np.sqrt(sigma11_sq).*np.sqrt(sigma12_sq) + C)
	rho2 = (sigma2_cross + C)./(np.sqrt(sigma21_sq).*np.sqrt(sigma22_sq) + C)
	C01map = 1 - 0.5*np.abs(rho1 - rho2)

	return C01map

def compute_C10_term(im1, im2, win):
	assert ((im1.dtype == float) and (im2.dtype == float))
	C = 0.001;
	window2 = 1/(win*(win-1)) * np.ones(win-1,win);

	im11 = im1[:, : end-1]
	im12 = im1[:, 1:]
	im21 = im2[:, :end-1]
	im22 = im2[:, 1:]

	mu11 = sc.convolve2d(im11, window2, mode = 'valid')
	mu12 = sc.convolve2d(im12, window2, mode = 'valid')
	mu21 = sc.convolve2d(im21, window2, mode = 'valid')
	mu22 = sc.convolve2d(im22, window2, mode = 'valid')

	sigma11_sq = sc.convolve2d(im11*im11, window2, mode = 'valid') - mu11*mu11
	sigma11_sq = sc.convolve2d(im12*im12, window2, mode = 'valid') - mu12*mu12
	sigma11_sq = sc.convolve2d(im21*im21, window2, mode = 'valid') - mu21*mu21
	sigma11_sq = sc.convolve2d(im22*im22, window2, mode = 'valid') - mu22*mu22

	sigma1_cross = sc.convolve2d(im11*np.conj(im12), window2, mode = 'valid') - mu11*np.conj(mu12)
	sigma2_cross = sc.convolve2d(im21*np.conj(im22), window2, mode = 'valid') - mu21*np.conj(mu12)

	rho1 = (sigma1_cross + C)./(np.sqrt(sigma11_sq).*np.sqrt(sigma12_sq) + C)
	rho2 = (sigma2_cross + C)./(np.sqrt(sigma21_sq).*np.sqrt(sigma22_sq) + C)
	C10map = 1 - 0.5*np.abs(rho1 - rho2)

	return C10map


# class Metric:

# 	def __init__():
# 		self.Nsc = 2
# 		self.Nor = 4

# 	def ssim(a,b):
		

	# def stsim2(a, b):
	# 	s = Steerable(height = self.Nsc)

	# 	pyrA = s.buildSFpyr(a)
	# 	pyrB = s.buildSFpyr(b)


