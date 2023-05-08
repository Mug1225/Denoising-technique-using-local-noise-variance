import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import scipy.signal as sig
import os
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def filterfun(img,block,w,h):
    dst = np.zeros((h, w), dtype=np.uint8)
    kernel1 = np.asarray([[1, 1, 1], [1, 8, 1], [1, 1, 1]]) / 16
    conv1 = sig.convolve2d(img, kernel1, mode='same')
    means = cv2.boxFilter(img, -1, tuple(block), normalize=True, borderType=cv2.BORDER_REPLICATE)

    kernel2=np.asarray([[1,2,1],[2,0,2],[1,2,1]])/12
    conv2=sig.convolve2d(img,kernel2,mode='same')
    variances = (conv1 - means) ** 2 / (block[0] * block[1])
    noiseVariance = cv2.boxFilter(variances, -1, tuple(block), normalize=True, borderType=cv2.BORDER_REPLICATE)
    for i in range(h):
        for j in range(w):
            #dst[i,j] = np.uint8(img[i,j] + max(0,variances[i,j]-noiseVariance[i,j])/avgnoiseVariance * (img[i,j]-means[i,j]))
            if variances[i, j] - noiseVariance[i, j]<0:
                dst[i,j]=img[i,j]
            else:
                dst[i,j]=conv2[i,j]
            #dst[i,j]=np.uint8(img[i, j] + max(0, variances[i, j] - noiseVariance[i, j]) / noiseVariance[i,j] * (conv2[i,j] - means[i, j]))

    return dst

if __name__== "__main__":
    root=Tk()
    root = Tk()
    root.withdraw()
    path=os.getcwd()
    path = filedialog.askopenfilename()
    image = cv2.imread(path)

    img = Image.open(path)
    m, n = img.size
    img = cv2.imread(path)
    img2=img.copy()
    kernel = np.array([3, 3]) //can be any odd  kernel for evaluating local noise
    img2[:, :, 0] = filterfun(img[:, :, 0], kernel, m, n)
    img2[:, :, 1] = filterfun(img[:, :, 1], kernel, m, n)
    img2[:, :, 2] = filterfun(img[:, :, 2], kernel, m, n)
    cv2.imwrite("noisefilter.jpg", img2)

    print("PSNR = ", peak_signal_noise_ratio(img, img2))
    print("MSE = ", mean_squared_error(img, img2))
    print("done")
