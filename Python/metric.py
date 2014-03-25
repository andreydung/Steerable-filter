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

def fspecial(shape = (11, 11), sigma = 1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh
    return h


# class Metric:

# 	def __init__():
# 		self.Nsc = 2
# 		self.Nor = 4

# 	def ssim(a,b):
		

	# def stsim2(a, b):
	# 	s = Steerable(height = self.Nsc)

	# 	pyrA = s.buildSFpyr(a)
	# 	pyrB = s.buildSFpyr(b)


