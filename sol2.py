import numpy as np
from scipy import signal
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

"""
1D discrete signal to its Fourier representation.
Signal is an array of dtype float64 with shape (N,1)
"""
def DFT(signal):
    N = signal.size
    k = np.arange(N)
    l = k.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * l / N)
    F = np.dot(e, signal)
    return F


"""
Function that transform a Fourier representation to its 1D discrete signal
Fourier_signal is an array of dtype complex128
"""
def IDFT(fourier_signal):
    N = fourier_signal.size
    k = np.arange(N)
    l = k.reshape((N, 1))
    e = np.exp(2j * np.pi * k * l / N)
    f = np.dot(e, fourier_signal) / N
    return f

"""
2-D Fourier Transform computed using 1-D Fourier.
Image is a grayscale image of dtype float64,
"""
def DFT2(image):
    F = []
    N, M = image.shape
    imageTranpose = image.T
    DFTcol = []
    # Compute 1-D Fourier on each column
    for col in range(M):
        DFTcol.append(DFT(imageTranpose[col]))

    # On result: Compute 1-D Fourier on each row
    DFTcolTranspose = np.array(DFTcol).T
    for row in range(N):
        F.append(DFT(DFTcolTranspose[row]))

    return np.asarray(F)


"""
Function that transform a Fourier representation to its 2D discrete signal.
fourier_image is a 2D array of dtype complex128.
"""
def IDFT2(fourier_image):
    f = []
    N, M = fourier_image.shape
    imageTranpose = fourier_image.T
    DFTcol = []
    # Compute 1-D Fourier on each column
    for col in range(M):
        DFTcol.append(IDFT(imageTranpose[col]))

    # On result: Compute 1-D Fourier on each row
    DFTcolTranspose = np.array(DFTcol).T
    for row in range(N):
        f.append(IDFT(DFTcolTranspose[row]))

    return np.asarray(f)


"""
Function that computes the magnitude of image derivatives. derives the image in each
direction separately (vertical and horizontal) using simple convolution with [1, 0, −1] as a row and column
vectors, to get the two image derivatives. Next, use these derivative images to compute the magnitude.
Where the input and the output are grayscale images of type float64
image.
"""
def conv_der(im):
    derive_x = np.asarray([[1, 0, -1]])
    derive_y = derive_x.reshape((3, 1))
    im_dx = signal.convolve2d(im, derive_x, mode="same")
    im_dy = signal.convolve2d(im, derive_y, mode="same")
    magnitude = np.sqrt(np.abs(im_dx)**2 + np.abs(im_dy)**2)

    return magnitude


"""
Finds the list we need to multiply the DFT before finding the derivative.
"""
def find_mul_lst(num):
    high_bound = int(np.ceil(num / 2))
    low_bound = - int(num / 2)
    return np.matrix([[i] for i in range(low_bound, high_bound)])


"""
Function that computes the magnitude of image derivatives using Fourier transform. Uses the
formula from class to derive in the x and y directions.
Where the input and the output are float64 grayscale images
"""
def fourier_der(im):
    F_im = DFT2(im)
    dft_im = np.fft.fftshift(F_im)
    N, M = im.shape
    u = find_mul_lst(N)
    v = find_mul_lst(M)
    mult_x = np.dot(u.T, dft_im)
    mult_y = np.dot(dft_im, v)
    dx = IDFT2(np.asarray(mult_x))
    dy = IDFT2(np.asarray(mult_y))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


"""
Function that derives a row of binomial coefficient.
"""
def gaussian_maker(kernel_size):
    gaussian = [1 / 2, 1 / 2]
    consequent = [1 / 2, 1 / 2]
    for i in range(kernel_size - 2):
        gaussian = signal.convolve(gaussian, consequent)
    gaussian_2D = signal.convolve(np.asarray(gaussian).reshape(1, kernel_size),
                                  np.asarray(gaussian).reshape(kernel_size, 1))
    return gaussian_2D


"""
Function that performs image blurring using 2D convolution between the image f and a gaussian
kernel g.
im - is the input image to be blurred (grayscale float64 image).
kernel_Size - is the size of the gaussian kernel in each dimension (an odd integer).
The function returns the output blurry image (grayscale float64 image).
"""
def blur_spatial(im, kernel_size):
    image_blurred = im
    if kernel_size > 1:
        g = gaussian_maker(kernel_size)
        image_blurred = signal.convolve2d(im, g, mode='same')
    return image_blurred


"""
Function that performs image blurring with gaussian kernel in Fourier space.
im - is the input image to be blurred (grayscale float64 image).
kernel_Size - is the size of the gaussian kernel in each dimension (an odd integer).
The function returns the output blurry image (grayscale float64 image).
"""
def blur_fourier(im, kernel_size):
    g = gaussian_maker(kernel_size)
    N, M = im.shape
    matrix = np.asarray(np.zeros((N, M)))
    row_center = int((N - kernel_size) / 2)
    col_center = int((M - kernel_size) / 2)
    matrix[row_center: (row_center + kernel_size), col_center: (col_center + kernel_size)] = g
    g = np.fft.ifftshift(matrix)
    G = DFT2(g)
    F = DFT2(im)
    F_dot_G = np.multiply(F, G)
    image_space = IDFT2(F_dot_G)
    return image_space


"""
function which reads an image file and converts it into a given representation.
This function returns an image, normalized to the range [0, 1].
representation = 1 for rgb picture, 2 for grayscale picture.
"""
def read_image(filename, representation):
    im = imread(filename).astype(np.float64) / 255
    if (representation == 1):
        im_g = rgb2gray(im)
        return im_g
    return im
