# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:33:09 2022

@author: Blumberg
"""
import numpy as np
from matplotlib import pyplot as plt

x_bounds = np.array([-1,1])
y_bounds = np.array([-2,2])

#Methode 1:
#wie von dir vorgeschlagen. Also Intervallgrenzen bestimmen, jeweils in n Teile
#teilen und dann ein symmetrisches Raster aufbauen, sodass n^N Basisvektoren
#entstehen. Mit N= Anzahl der Inputs

n = 4
x_grid = np.linspace(x_bounds[0], x_bounds[1], n)
y_grid = np.linspace(y_bounds[0], y_bounds[1], n)

basis_vectors_1 = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1,2)

fig, axs = plt.subplots(1,3)
axs[0].scatter(basis_vectors_1[:,0], basis_vectors_1[:,1])

#Methode 2:
#Der Nachteil von Methode 1 ist meiner Meinung nach (ganz intuitiv gedacht,
#ohne das auf irgendwelche Beweise zu stützen), dass die Basisvektoren zwar 
#schön gleichmäßig im Raum verteilt sind, es für die Inputs x und y aber jeweils
#nur n unterschiedliche Werte gibt und diese sich oft wiederholen, auch
#wenn die Kombination jedesmal eine neue ist.
#Bei gleicher Anzahl an Basisvektoren könnte man auch die Intervalle jeweils
#in eben diese Anzahl einteilen

nn = n**2
x_grid = np.linspace(x_bounds[0], x_bounds[1], nn)
y_grid = np.linspace(y_bounds[0], y_bounds[1], nn)

basis_vectors_2 = np.vstack((x_grid, y_grid)).T
axs[1].scatter(basis_vectors_2[:,0], basis_vectors_2[:,1])

#Methode 3:
#Bei der Methode 2 ist offensichtlich der Nachteil, dass die Basisvektoren
#schlecht verteilt sind, weil sie auf der Diagonalen liegen. Aber es gibt
#mehr verschiedene Einträge je Achse und auch die Anzahl der Basisvektoren
#lässt sich besser, nämlich komplett individuell wählen.
#Jetzt wird die Kombination der Verteilungen je input noch zufällig gewählt

np.random.shuffle(x_grid)
np.random.shuffle(y_grid)

basis_vectors_3 = np.vstack((x_grid, y_grid)).T

basis_vectors_3 = np.vstack((x_grid, y_grid)).T
axs[2].scatter(basis_vectors_3[:,0], basis_vectors_3[:,1])

