import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.image as mpimg
import tempfile
import skimage.io as skio
import os
from math import exp, floor 
from scipy import signal

def convolGrain(im, sigma, k):
    
    imt = normalize(im)
    sh = imt.shape
    
    #creation du filtre gaussien
    filter_size = 9
    sigma = 1.0
    h_filter_size = int(np.floor(filter_size/2.0))
    varfiltre = np.zeros((filter_size, filter_size))
    
    for x in range(-h_filter_size, h_filter_size+1):
        for y in range(-h_filter_size, h_filter_size+1):
            varfiltre[y+h_filter_size, x+h_filter_size] = gauss(x,y,sigma)
            
    #normaliser filtre
    varfiltre = 1/(np.sum(varfiltre)) * varfiltre
    
    noise_out = signal.convolve2d(np.random.randn(sh[0]+8,sh[1]+8), varfiltre,mode='valid',boundary='wrap')
        
    return imt + k*noise_out

def convolGrainC(im, sigma, k):
    sh = (im.shape[0], im.shape[1])
    
    #on utilise la méthode précédente sur chacun des canaux de couleur
    imrouge = im[:,:,0]
    imvert = im[:,:,1]
    imbleu = im[:,:,2]
    
    rouge = convolGrain(imrouge, sigma, k)
    vert = convolGrain(imvert, sigma, k)
    bleu = convolGrain(imbleu, sigma, k)
    
    #concaténation des 3 canaux
    newim = [[0]*sh[1] for _ in range(sh[0])]
    for i in range(sh[0]):
        for j in range(sh[1]):
            newim[i][j] = (rouge[i][j], vert[i][j], bleu[i][j])

    return newim
    

#fonction ajoutant un grain sur une image en niveaux de gris
def addGrain(im, sigma, k):
    imt = normalize(im)
    sh = imt.shape
    
    #on crée un bruit blanc, puis passage en fréquentiel
    bruit = noise(sh)
    noisefft = fft2(bruit)
    
    #on crée un filtre gaussien, puis passage en fréquentiel
    filtre = filtreGauss(sh, sigma)
    filtrefft = fft2(filtre)
    
    #bruit final
    filteredNoise = noisefft*filtrefft
    
    return imt + k*ifft2(filteredNoise)


def addGrainC(im, sigma, k):
    
    sh = (im.shape[0], im.shape[1])
    
    #on utilise la méthode précédente sur chacun des canaux de couleur
    imrouge = im[:,:,0]
    imvert = im[:,:,1]
    imbleu = im[:,:,2]
    
    rouge = addGrain(imrouge, sigma, k)
    vert = addGrain(imvert, sigma, k)
    bleu = addGrain(imbleu, sigma, k)
    
    #concaténation des 3 canaux
    newim = [[0]*sh[1] for _ in range(sh[0])]
    for i in range(sh[0]):
        for j in range(sh[1]):
            newim[i][j] = (rouge[i][j], vert[i][j], bleu[i][j])

    return newim

def filtreGauss(sh, sigma):
    #crée un filtre gaussien de paramètre sigma de taille sh
    filtre = [[0]*sh[1] for _ in range(sh[0])]
    for i in range(sh[0]):
        for j in range(sh[1]):
            x = i - floor(sh[0]/2)
            y = j - floor(sh[1]/2)
            filtre[i][j] = gauss(x, y, sigma)
    
    normalize(filtre)
    
    return filtre


def noise(sh):
    #crée un bruit blanc gaussien de taille sh
    return np.random.randn(*sh)

def gauss(x, y, sigma):
    return exp(-(x**2 + y**2)/(2*sigma**2))

def viewimage(im):
    #fonction affichant une image dans gimp
    prephrase='gimp '
    endphrase= ' &'
    imt = normalize(im)

    nomfichier = tempfile.mktemp('projet.png')
    commande = prephrase + nomfichier + endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)
    
    return

def normalize(im, normalize = True, MINI = 0.0, MAXI = 255.0):
    imt = np.float32(im.copy())
    if normalize:
        m = imt.min()
        imt = imt-m
        M = imt.max()
        if M > 0:
            imt = imt/M

    else:
        imt = (imt-MINI)/(MAXI-MINI)
        imt[imt < 0] = 0
        imt[imt > 1] = 1
   
    return imt



