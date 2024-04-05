import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Sumo():

    def __init__(self, scenario):
        u = self.load_csv('sumo/'+scenario+'/spaciotemporal.csv')  # density(time, position) (1000, 300)
        self.L, self.Tmax = float(u[0][0]), float(u[0][1])
        self.u = np.array(u[1:]).astype(np.float)
        
        data_train = self.load_csv('sumo/'+scenario+'/pv.csv')  # (measurements, features) features=(position, time, density, speed)
        data_train = np.array(data_train).astype(np.float)
        self.probe_t, self.probe_x, self.probe_u, self.probe_v = self.process_probe_data(data_train)  # probe_density(vehicle, position, time, density)

        self.Nx, self.Nt = self.u.shape


    def load_csv(self, file):
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data

    def process_probe_data(self, data):
        probe_x = [] # position
        probe_t = [] # time
        probe_u = [] # density
        probe_v = [] # speed
        
        pv_x = []
        pv_t = []
        pv_u = []
        pv_v = []
        
        t_prev = 0
        for meas in data:
            t = meas[1]
            if t - t_prev < 0:
                probe_x.append(np.array(pv_x).reshape((-1,1)))
                pv_x = []
                probe_t.append(np.array(pv_t).reshape((-1,1)))
                pv_t = []
                probe_u.append(np.array(pv_u).reshape((-1,1)))
                pv_u = []
                probe_v.append(np.array(pv_v).reshape((-1,1)))
                pv_v = []
            pv_x.append(meas[0])
            pv_t.append(meas[1])
            pv_u.append(meas[2])
            pv_v.append(meas[3])
            t_prev = t
        probe_x.append(np.array(pv_x).reshape((-1,1)))
        probe_t.append(np.array(pv_t).reshape((-1,1)))
        probe_u.append(np.array(pv_u).reshape((-1,1)))
        probe_v.append(np.array(pv_v).reshape((-1,1)))
        
        return probe_t, probe_x, probe_u, probe_v

    def getMeasurements(self):
        return self.probe_t, self.probe_x, self.probe_u, self.probe_v

    def getDensity(self):
        return self.u
    
    def getAxisPlot(self):
        t = np.linspace(0, self.Tmax, self.Nt) # Nt is the number of temporal points
        x = np.linspace(0, self.L, self.Nx) # Nx is the number of spatial points
        return (x, t)

    def plotDensity(self):
        fig = plt.figure('density_true', figsize=(7.5, 5))
        plt.imshow(np.flipud(self.u), extent=[0, self.Tmax, 0, self.L], 
                   cmap='rainbow', vmin=0.0, vmax=1, aspect='auto')
        plt.colorbar()
        # for (t,x) in zip(self.probe_t, self.probe_x):
        #     plt.plot(t, x, c='k')
        # plt.title('Density')
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.ylim(0, self.L)
        plt.xlim(0, self.Tmax)
        plt.tight_layout()
        fig.savefig('density.eps', bbox_inches='tight')

    def plotProbeVehicles(self):
        for (t,x) in zip(self.probe_t, self.probe_x):
            plt.plot(t, x, c='k')

    def plotProbeDensity(self):
        for (t,x,u) in zip(self.probe_t, self.probe_x, self.probe_u):
            # plt.plot(t, x, c='k')
            plt.scatter(t, x, c=u, cmap='rainbow', vmin=0.0, vmax=1, s=5)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.ylim(0, self.L)
        plt.xlim(0, self.Tmax)
        plt.tight_layout()
        plt.colorbar()
        plt.show()

    

    def plotDensityMovieHeatmap(self):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 1))
        # Placeholder for the heatmap
        heatmap = ax.imshow(np.flipud(self.u[:, 0][:, np.newaxis].T), extent=[0, 420, 0, 250], cmap='inferno', vmin=0.0, vmax=1, aspect='auto')

        ax.set_xlabel('Position [km]')
        ax.set_yticks([])  # Remove ticks on y-axis
        ax.set_xticks([]) 
        # ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])  # Set ticks on x-axis
        time_text = ax.text(0.83, 0.85, '', transform=ax.transAxes, color='white')  # Placeholder for time annotation

        # Initialization function for the animation
        def init():
            heatmap.set_data(np.flipud(self.u[:, 0][:, np.newaxis]))
            return heatmap,

        # Update function for each frame
        def update(frame):
        # Ensuring we pass a 2D array for each frame
            current_data = self.u[:, frame]
            if current_data.ndim == 1:
                current_data = current_data[:, np.newaxis]
            heatmap.set_data(np.flipud(current_data).T)  # Update the heatmap for the current frame
            time_text.set_text(f'Time: {frame:.2f} s')  # Update time annotation
            return heatmap, time_text

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=int(420), init_func=init, blit=True)

        # Save the animation in transparant
        ani.save('heatmap_animation.gif', writer='pillow', fps=50, dpi=400, savefig_kwargs={'transparent': True})


    def plotDensityMovie(self):
        # Setting up the figure for the animation
        breakpoint()
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], lw=2)  # Initialize an empty line plot
        ax.set_xlim(0, 249)  # Assuming x-axis represents the position, adjust as needed
        ax.set_ylim(0, np.max(self.u))  # Adjust y-axis limits based on your data range
        ax.set_xlabel('Position [km]')
        ax.set_ylabel('Amplitude')
        time_text = ax.text(0.83, 0.85, '', transform=ax.transAxes)  # Placeholder for time annotation
        self.u = np.flipud(self.u)

        # Initialization function for the animation
        def init():
            line.set_data([], [])
            return line,

        # Update function for each frame
        def update(frame):
            line.set_data(range(250), self.u[:, frame])  # Update the line plot for the current frame
            time_text.set_text(f'Time: {frame:.2f} s')  # Update time annotation
            return line, time_text

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=420, init_func=init, blit=True)

        # Save the animation
        ani.save('lineplot_animation.gif', writer='pillow', fps=50, dpi=50)

    
    def plotProbeFD(self):
        fig = plt.figure('FD', figsize=(7.5, 5))
        for (t,x,u,v) in zip(self.probe_t, self.probe_x, self.probe_u, self.probe_v):
            plt.scatter(u, v, c='k', s=5)
        plt.xlabel(r'Density [veh/km]')
        plt.ylabel(r'Speed [km/min]')
        plt.tight_layout()
