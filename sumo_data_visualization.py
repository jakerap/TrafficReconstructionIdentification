import numpy as np
np.random.seed(12345)
# import reconstruction_neural_network as rn
import reconstruction_neural_network_pytorch as rn
import sumo as s
import matplotlib.pyplot as plt

scenario = 'highway'

sumo = s.Sumo(scenario)
Nx, Nt = sumo.Nx, sumo.Nt # Number of spatial and temporal points (250, 420)
L, Tmax = sumo.L, sumo.Tmax # Length of the road and maximum time (2.5, 7.0)

rho = sumo.getDensity()  # density(time, position) (250, 420)
# sumo.plotDensity()
# plt.show()
# sumo.plotProbeDensity()
t_train, x_train, rho_train, v_train = sumo.getMeasurements()
Vf = np.amax(v_train[14]) # find the highest speed in the training data

net = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train, L, Tmax, N_f=1000, N_g=500, opt=9, v_max=Vf)
t_train, x_train, u_train, v_train, X_f_train, t_g_train, u_v_train, v_max = net.createTrainingDataset(t_train, x_train, rho_train, v_train, L, Tmax, N_f=1000, N_g=500, N_v=50, Tmax=Tmax)

x_f_train = X_f_train[:, 0:1] # space data points for the PDE [500, 1]
t_f_train = X_f_train[:, 1:2] # time data points for the PDE [500, 1]

# scatter plot of all training data locations for density
# plt.figure()
# plt.scatter(x_f_train, t_f_train, c='b', s=1)
# for i in range(len(t_train)):
#     plt.scatter(t_train[i], x_train[i], c='r', s=3)
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Training data locations')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.grid()


# plt.figure()
# plt.plot(u_v_train, np.zeros_like(u_v_train), 'rx')
# plt.ylabel('Velocity')
# plt.title('Training data locations')
# # plt.xlim([-1, 1])
# # plt.ylim([-1, 1])
# plt.grid()


# fig = plt.figure('FD', figsize=(7.5, 5))
# colors = plt.cm.rainbow(np.linspace(0, 1, len(u_train)))

# for (u, v, color) in zip(u_train, v_train, colors):
#     plt.scatter(u, v, color=color, s=5)

# plt.xlabel(r'Density [veh/km]')
# plt.ylabel(r'Speed [km/min]')
# plt.tight_layout()
# plt.show()



# sumo.plotProbeVehicles()
sumo.plotProbeFD()
plt.show()

# sumo.plotDensityMovieHeatmap()
# sumo.plotProbes()
# sumo.plotDensityMovie()
# sumo.plotDensityMovie()
# sumo.plotCombinedDensityMovie()
sumo.plotDensityAtTimePoints()