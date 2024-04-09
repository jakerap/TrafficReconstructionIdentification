# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import godunov as g
import reconstruction_neural_network as rn
from pyDOE import lhs
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#####################################
####     General parameters     #####
#####################################

Vf = 1.5 # Maximum car speed in km.min^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small but not zero)
Tmax = 2 # simulation time in min
p = 1/20 # Probability that a car is a PV
L = 5 # Length of the road in km
rhoBar = 0.5 # Average density of cars on the road
rhoSigma = 0.45 # initial condition standard deviation
rhoMax = 120 # Number of vehicles per kilometer
noise = True # noise on the measurements and on the trajectories
greenshield = True # Type of flux function used for the numerical simulation
Ncar = rhoBar*rhoMax*L # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

t0 = time.time()

# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=L, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=500, greenshield=greenshield,
                             rhoBar=rhoBar, rhoSigma=rhoSigma)

rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)

t1 = time.time()
print("simulation duration: ", t1-t0, " seconds")


def plotProbeFD(t_train, x_train, rho_train, v_train):
      fig = plt.figure('FD', figsize=(7.5, 5))
      for (t,x,u,v) in zip(t_train, x_train, rho_train, v_train):
            plt.scatter(u, v, c='k', s=5)
            plt.xlabel(r'Density [veh/km]')
            plt.ylabel(r'Speed [km/min]')
            plt.tight_layout()
      plt.show()


def plotDensityAtTimePoints(rho):
      fig, ax = plt.subplots(1, 3, figsize=(15, 5))
      for i, t in enumerate([0, 210, 374]):
            ax[i].plot(rho[:, t])
            ax[i].set_title(f'Time: {t}')
            plt.tight_layout()
      plt.show()


def plotCombinedDensityMovie(t_train, x_train, rho_train, v_train, rho):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Placeholder for the heatmap
        data = rho[:, 0][:, np.newaxis].T
        heatmap = ax.imshow(data, extent=[0, 500-1, 0, np.max(rho)], cmap='rainbow', vmin=0.0, vmax=1, aspect='auto')
        
        # Initialize an empty line plot for signal overlay
        line, = ax.plot([], [], lw=2, color='white')  # Using white for visibility against the heatmap
        
        # ax.set_ylabel('Amplitude')
      #   ax.set_xlim(0, 249)
      #   ax.set_ylim(0, np.max(rho))

        ax.set_yticks([])  # Remove ticks on y-axis
        # ax.set_xticks([]) 
      #   tick_positions = np.linspace(0, 250, 6)  # Generates 6 positions from 0 to 249 (inclusive)
      #   tick_labels = [str(int(label)) for label in np.linspace(0, 2500, 6)]  # Generates labels from 0 to 2490

      #   # Apply tick positions and labels to the axis
      #   ax.set_xticks(tick_positions)
      #   ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Position [m]')

        plt.tight_layout()
        
        # Placeholder for time annotation, positioned to be visible against the heatmap
        time_text = ax.text(0.87, 0.85, '', transform=ax.transAxes, color='black')
        
        # Initialize probe lines for each probe vehicle, making them visible from the start
        probe_lines = [ax.axvline(x[0], lw=3, color='black', visible=False) for x in x_train]  # Cyan for visibility
        probe_circles = [ax.scatter(x[0], rho_train[0][0], color='black', visible=False) for x in x_train]  # Red for visibility

        # Initialization function for the animation
        def init():
            heatmap.set_data(rho[:, 0][:, np.newaxis])
            line.set_data([], [])
            for pline, pcircle in zip(probe_lines, probe_circles):
                pline.set_visible(False)
                pcircle.set_visible(False)
            return [heatmap, line] + probe_lines + probe_circles

        # Update function for each frame
        def update(frame):
            # Updating the heatmap
            current_data = rho[:, frame]
            if current_data.ndim == 1:
                current_data = current_data[:, np.newaxis]
            heatmap.set_data(current_data.T)
            
            # Updating the line plot for the current frame
            line.set_data(range(500), rho[:, frame])
            
            # Updating time annotation
            time_text.set_text(f'Time: {frame/185.5:.2f} min')  # Assuming frame represents seconds, convert to minutes
            
            # Update probe lines based on current frame, checking against probe times
            for pv in range(len(probe_lines)):
                # probe_lines[pv].set_visible(False)  # Initially hide each line, only show if corresponding time matches
                for i, t in enumerate(t_train[pv]):
                    if int(t*185.5) == frame:  # Assuming t is in minutes, convert to frames
                        x_value = x_train[pv][i]*100
                        y_value = rho_train[pv][i]
                       
                        probe_circles[pv].set_offsets([[x_value, y_value]])
                        probe_lines[pv].set_data([x_value, x_value], [0, y_value])
                        probe_circles[pv].set_visible(True)
                        probe_lines[pv].set_visible(True)
                        
            
            return [heatmap, line, time_text] + probe_lines + probe_circles
        

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=375, init_func=init, blit=True)

        # Save the animation
        ani.save('combined_animation_godunov.gif', writer='pillow', fps=30, dpi=300, savefig_kwargs={'transparent': True})


def plotProbes(t_train, x_train, rho_train, v_train):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 

        # Plot Probe Density
        for (t, x, u) in zip(t_train, x_train, rho_train):
            sc1 = ax1.scatter(t, x, c=u, cmap='rainbow', vmin=0.0, vmax=1, s=5)
        ax1.set_xlabel(r'Time [min]')
        ax1.set_ylabel(r'Position [km]')
      #   ax1.set_ylim(0, self.L)
      #   ax1.set_xlim(0, self.Tmax)
        ax1.set_title('Probe Density')
        fig.colorbar(sc1, ax=ax1, label='Density')

        # Plot Probe Speed
        for (t, x, v) in zip(t_train, x_train, v_train):
            sc2 = ax2.scatter(t, x, c=v, cmap='rainbow', s=5)
        ax2.set_xlabel(r'Time [min]')
        ax2.set_ylabel(r'Position [km]')
      #   ax2.set_ylim(0, self.L)
      #   ax2.set_xlim(0, self.Tmax)
        ax2.set_title('Probe Speed')
        fig.colorbar(sc2, ax=ax2, label='Speed')

        plt.tight_layout()
        plt.savefig('probes_density_velocity.png', bbox_inches='tight')
        plt.show()


def plotDensityMovie(rho, t_train, x_train, rho_train, v_train):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], lw=2)  # Initialize an empty line plot
      #   ax.set_xlim(0, 249)  # Assuming x-axis represents the position, adjust as needed
        ax.set_ylim(0, np.max(rho))  # Adjust y-axis limits based on your data range
        ax.set_xlabel('Position [km]')
        ax.set_ylabel('Amplitude')
        time_text = ax.text(0.83, 0.85, '', transform=ax.transAxes)  # Placeholder for time annotation

        probe_lines = [ax.axvline(x[0]*100, lw=1, color='red', visible=False) for x in x_train] 

        # Initialization function for the animation
        def init():
            line.set_data([], [])
            for pline in probe_lines:
                pline.set_visible(False)  # Initially set all probe lines to be invisible
            return [line] + probe_lines

        # Update function for each frame
        def update(frame):
            line.set_data(range(500), rho[:, frame])  # Update the line plot for the current frame
            time_text.set_text(f'Time: {frame:.2f} s')  # Update time annotation
            for pv in range(len(probe_lines)):
                for i, t in enumerate(t_train[pv]):
                    if (int(t*187.5) == frame):
                        probe_lines[pv].set_data([x_train[pv][i]*100, x_train[pv][i]*100], [0,rho_train[pv][i]])
                        probe_lines[pv].set_visible(True)                    
                
            return [line, time_text]# + probe_lines

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=375, init_func=init, blit=True)

        # Save the animation
        ani.save('lineplot_animation_godunov.gif', writer='pillow', fps=50, dpi=50)


# rho [Nx, Nt], [500, 375]
# breakpoint()

# plotProbeFD(t_train, x_train, rho_train, v_train)
# plotDensityAtTimePoints(rho)
# plotProbes(t_train, x_train, rho_train, v_train)
plotCombinedDensityMovie(t_train, x_train, rho_train, v_train, rho)
# breakpoint()
# plotDensityMovie(rho, t_train, x_train, rho_train, v_train)
