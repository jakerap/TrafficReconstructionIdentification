# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import reconstruction_neural_network as rn
import sumo as s
import matplotlib.pyplot as plt

#####################################
####     General parameters     #####
#####################################

scenario = 'highway'

sumo = s.Sumo(scenario)
Nx, Nt = sumo.Nx, sumo.Nt # Number of spatial and temporal points (250, 420)
L, Tmax = sumo.L, sumo.Tmax # Length of the road and maximum time (2.5, 7.0)

rho = sumo.getDensity()  # density(time, position) (250, 420)
sumo.plotDensity()
t_train, x_train, rho_train, v_train = sumo.getMeasurements()
axisPlot = sumo.getAxisPlot()
Vf = np.amax(v_train[14]) # find the highest speed in the training data

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, N_f=500, N_g=50, opt=9)
trained_neural_network.start()
trained_neural_network.train()

# [_, _, figError] = trained_neural_network.plot(axisPlot, rho)
# sumo.plotProbeVehicles()
# figError.savefig('error.eps', bbox_inches='tight')
# trained_neural_network.close()


[_, _, figError] = trained_neural_network.plot(axisPlot, rho)
sumo.plotProbeVehicles()
figError.savefig('error.png', bbox_inches='tight')  # Changed file extension to .png
plt.show()
trained_neural_network.close()