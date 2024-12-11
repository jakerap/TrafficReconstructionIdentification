import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
# import plotly.express as px
from scipy.interpolate import griddata




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


    def plotProbeSpeed(self):
        for (t,x,v) in zip(self.probe_t, self.probe_x, self.probe_v):
            # plt.plot(t, x, c='k')
            plt.scatter(t, x, c=v, cmap='rainbow', s=5)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.title('Probe Speed')  # Add title
        plt.ylim(0, self.L)
        plt.xlim(0, self.Tmax)
        plt.tight_layout()
        plt.colorbar()
        plt.show()


    def plotProbes(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 

        # Plot Probe Density
        for (t, x, u) in zip(self.probe_t, self.probe_x, self.probe_u):
            sc1 = ax1.scatter(t, x, c=u, cmap='rainbow', vmin=0.0, vmax=1, s=5)
        ax1.set_xlabel(r'Time [min]')
        ax1.set_ylabel(r'Position [km]')
        ax1.set_ylim(0, self.L)
        ax1.set_xlim(0, self.Tmax)
        ax1.set_title('Probe Density')
        fig.colorbar(sc1, ax=ax1, label='Density')

        # Plot Probe Speed
        for (t, x, v) in zip(self.probe_t, self.probe_x, self.probe_v):
            sc2 = ax2.scatter(t, x, c=v, cmap='rainbow', s=5)
        ax2.set_xlabel(r'Time [min]')
        ax2.set_ylabel(r'Position [km]')
        ax2.set_ylim(0, self.L)
        ax2.set_xlim(0, self.Tmax)
        ax2.set_title('Probe Speed')
        fig.colorbar(sc2, ax=ax2, label='Speed')

        plt.tight_layout()
        plt.savefig('probes_density_velocity.png', bbox_inches='tight')
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
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], lw=2)  # Initialize an empty line plot
        ax.set_xlim(0, 249)  # Assuming x-axis represents the position, adjust as needed
        ax.set_ylim(0, np.max(self.u))  # Adjust y-axis limits based on your data range
        ax.set_xlabel('Position [km]')
        ax.set_ylabel('Amplitude')
        time_text = ax.text(0.83, 0.85, '', transform=ax.transAxes)  # Placeholder for time annotation
        # self.u = np.flipud(self.u)

        probe_lines = [ax.axvline(x[0]*100, lw=1, color='red', visible=False) for x in self.probe_x] 

        # Initialization function for the animation
        def init():
            line.set_data([], [])
            for pline in probe_lines:
                pline.set_visible(False)  # Initially set all probe lines to be invisible
            return [line] + probe_lines

        # Update function for each frame
        def update(frame):
            line.set_data(range(250), self.u[:, frame])  # Update the line plot for the current frame
            time_text.set_text(f'Time: {frame:.2f} s')  # Update time annotation
            for pv in range(len(probe_lines)):
                for i, t in enumerate(self.probe_t[pv]):
                    if (int(t*60) == frame):
                        probe_lines[pv].set_data([self.probe_x[pv][i]*100, self.probe_x[pv][i]*100], [0,self.probe_u[pv][i]])
                        probe_lines[pv].set_visible(True)                    
                
            return [line, time_text] + probe_lines

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=420, init_func=init, blit=True)

        # Save the animation
        ani.save('lineplot_animation.gif', writer='pillow', fps=50, dpi=50)

    def plotCombinedDensityMovie(self):
        # Setting up the figure for the animation
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Placeholder for the heatmap
        data = self.u[:, 0][:, np.newaxis].T
        heatmap = ax.imshow(data, extent=[0, 249, 0, np.max(self.u)], cmap='rainbow', vmin=0.0, vmax=1, aspect='auto')
        
        # Initialize an empty line plot for signal overlay
        line, = ax.plot([], [], lw=2, color='white')  # Using white for visibility against the heatmap
        
        # ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 249)
        ax.set_ylim(0, np.max(self.u))

        ax.set_yticks([])  # Remove ticks on y-axis
        # ax.set_xticks([]) 
        tick_positions = np.linspace(0, 250, 6)  # Generates 6 positions from 0 to 249 (inclusive)
        tick_labels = [str(int(label)) for label in np.linspace(0, 2500, 6)]  # Generates labels from 0 to 2490

        # Apply tick positions and labels to the axis
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Position [m]')

        plt.tight_layout()
        
        # Placeholder for time annotation, positioned to be visible against the heatmap
        time_text = ax.text(0.87, 0.85, '', transform=ax.transAxes, color='black')
        
        # Initialize probe lines for each probe vehicle, making them visible from the start
        probe_lines = [ax.axvline(x[0]*100, lw=3, color='black', visible=False) for x in self.probe_x]  # Cyan for visibility
        probe_circles = [ax.scatter(x[0]*100, self.probe_u[0][0], color='black', visible=False) for x in self.probe_x]  # Red for visibility

        # Initialization function for the animation
        def init():
            heatmap.set_data(self.u[:, 0][:, np.newaxis])
            line.set_data([], [])
            for pline, pcircle in zip(probe_lines, probe_circles):
                pline.set_visible(False)
                pcircle.set_visible(False)
            return [heatmap, line] + probe_lines + probe_circles

        # Update function for each frame
        def update(frame):
            # Updating the heatmap
            current_data = self.u[:, frame]
            if current_data.ndim == 1:
                current_data = current_data[:, np.newaxis]
            heatmap.set_data(current_data.T)
            
            # Updating the line plot for the current frame
            line.set_data(range(250), self.u[:, frame])
            
            # Updating time annotation
            time_text.set_text(f'Time: {frame/60:.2f} min')  # Assuming frame represents seconds, convert to minutes
            
            # Update probe lines based on current frame, checking against probe times
            for pv in range(len(probe_lines)):
                # probe_lines[pv].set_visible(False)  # Initially hide each line, only show if corresponding time matches
                for i, t in enumerate(self.probe_t[pv]):
                    if int(t*60) == frame:  # Assuming t is in minutes, convert to frames
                        x_value = self.probe_x[pv][i]*100
                        y_value = self.probe_u[pv][i]
                        probe_lines[pv].set_data([x_value, x_value], [0, y_value])
                        probe_circles[pv].set_offsets([[x_value, y_value]])
                        probe_lines[pv].set_visible(True)
                        probe_circles[pv].set_visible(True)
            
            return [heatmap, line, time_text] + probe_lines + probe_circles
        


        # Creating the animation
        ani = FuncAnimation(fig, update, frames=420, init_func=init, blit=True)

        # Save the animation
        ani.save('combined_animation.gif', writer='pillow', fps=30, dpi=300, savefig_kwargs={'transparent': True})

    
    def plotProbeFD(self):
        fig = plt.figure('FD', figsize=(7.5, 5))
        for (t,x,u,v) in zip(self.probe_t, self.probe_x, self.probe_u, self.probe_v):
            plt.scatter(u, v, c='k', s=5)
        plt.xlabel(r'Density [veh/km]')
        plt.ylabel(r'Speed [km/min]')
        plt.tight_layout()


    def plotDensityAtTimePoints(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, t in enumerate([0, 210, 420-1]):
            ax[i].plot(self.u[:, t])
            ax[i].set_title(f'Time: {t}')
        plt.tight_layout()
        plt.show()


    def plotDensity3D(self):
        # breakpoint()
        time = np.linspace(0, 420, 420)
        space = np.linspace(0, 250, 250)

        # Create a meshgrid for space and time
        space_grid, time_grid = np.meshgrid(space, time)

        # Create a figure and a 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the surface plot
        surf = ax.plot_surface(space_grid, time_grid, self.u.T, cmap='rainbow')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title('Density Surface Plot')
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Density')

        # Show plot
        plt.show()

    def plotDensity3Dplotly(self):
        # breakpoint()
        time = np.linspace(0, 420, 420)
        space = np.linspace(0, 250, 250)

        # Create a meshgrid for space and time
        space_grid, time_grid = np.meshgrid(space, time)

        fig = go.Figure(data=[go.Surface(z=self.u, x=time_grid, y=space_grid, colorscale='Viridis')])

        # Update plot layout
        fig.update_layout(
            title='Density Surface Plot',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Space',
                zaxis_title='Density'
            ),
            autosize=False,
            width=800,
            height=600
        )

        # Show plot
        fig.show()