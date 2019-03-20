import numpy as np
import matplotlib.image as mpimg
from math import pi, sqrt
import cv2 as cv
import tempfile
import skimage.io as skio
import os
import pdb
import random
import scipy.misc
import scipy
import matplotlib.pyplot as plt

epsilon = 0.00001
rmax = 0.5
s = 5.0 #choix du zoom
monteCarlo = 100

def booleanModel(im, N, M):
    
    centres = []
    rayons = []
    
    np.random.seed( 3 )

    for i in range(N):
        for j in range(M):
            Uij = im[i][j]/(255.0 + epsilon) #étape de normalisation
            
            rmoyen = (epsilon+rmax)/2

            lmbda = (1/(pi*rmoyen**2))*np.log(1/(1-Uij)) #calcul du lambda
            Q = np.random.poisson(lmbda, 1)[0] #nombre de centres à tirer

            for k in range(Q): #tirage des Q centres et des Q rayons
                x = np.random.uniform(i, i+1)
                y = np.random.uniform(j, j+1)
                r = np.random.uniform(epsilon, rmax)
                centres.append([x, y])
                rayons.append([r])
                
    return centres, rayons

def evaluation(im):

    (N, M) = im.shape
    centres, rayons = booleanModel(im, N, M) 
    Q = len(centres)

    N_zoom = int(s*N)
    M_zoom = int(s*M)
    imf = np.zeros((N_zoom,M_zoom))

    rayons = np.asarray(rayons)
        
    for i in range(N_zoom): #pour chaque pixel final, je translate le centre de (e0, e1) et je cherche un disque dans centres assez proche
        for j in range(M_zoom):
            print(i,j)
            for k in range(monteCarlo):
                e0 = np.random.normal(0, 1) #tirage du vecteur translation
                e1 = np.random.normal(0, 1)
                i_s = i/s + 1/(2*s)
                j_s = j/s + 1/(2*s)
                x = i_s + e0/s
                y = j_s + e1/s
                for l in range(Q):
                    dist = (centres[l][0]-x)**2 + (centres[l][1]-y)**2
                    if dist <= rayons[l]*rayons[l]:
                        imf[i][j] += 1
                        break
    
    for i in range(N_zoom): #boucle pour normaliser, j'avais un problème avec imf/monteCarlo
        for j in range(M_zoom):
            imf[i][j] = float(imf[i][j])/monteCarlo
    
    return imf

def viewimage(im):
    #fonction affichant une image en niveaux de gris dans gimp
    prephrase='gimp '
    endphrase= ' &'

    nomfichier = tempfile.mktemp('projet.png')
    commande = prephrase + nomfichier + endphrase
    skio.imsave(nomfichier,im)
    os.system(commande)
    
    return
            
