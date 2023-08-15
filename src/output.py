# module output
'''
Manages output.
'''

import scienceplots # pylint: disable=unused-import
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from initializer import CharPoint
import constants as cn
import lines

def plotting(wall_data: tuple[list[float], list[float]],
             char_data: tuple[list[float], list[float]],
             calcd_area_ratio: float, ideal_area_ratio: float,
             percent_error: float, char_pts: list['CharPoint']):
    '''
    Plotting interface for a visual representation of the nozzle geometry.

    Args:
        wall_data (tuple[list[float], list[float]]): (x, y) coordinates for points on the wall
        char_data (tuple[list[float], list[float]]): (x, y) coordinates for all other points
        calcd_area_ratio (float): calculated area ratio using throat radius and final y coordinate
        ideal_area_ratio (float): ideal area ratio calculated using isentropic relations
        percent_error (float): % error between the calculated and ideal area ratios
        char_pts (list['CharPoint']): list of characteristic points
    '''

    # Use the scienceplots module and a dark theme
    plt.style.use(['science', 'grid', 'dark_background'])

    # Lines connecting the top and bottom wall points
    plt.plot(wall_data[0], wall_data[1], 'w')
    plt.plot(wall_data[0], [-y for y in wall_data[1]], 'w')

    # Top wall points
    plt.scatter(wall_data[0], wall_data[1], facecolors='none', edgecolors='w')

    # Interior and centerline points
    plt.scatter(char_data[0], char_data[1], facecolors='none', edgecolors='w')

    # Store the indices of the points that lie on the characteristics
    rght_chars = [lines.find_rght_chars(num) for num in range(2, cn.N_LINES + 1)]
    left_chars =  lines.find_left_chars(cn.N_LINES + 1)

    # Plot left-running characteristic lines
    for i in range(len(char_pts) - 1):
        for j, _ in enumerate(left_chars):
            # Ensure that the separate lines are not connected
            if char_pts[i].idx in left_chars[j] and char_pts[i+1].idx in left_chars[j]:
                plt.plot([char_pts[i].xy_loc[0], char_pts[i+1].xy_loc[0]],
                         [char_pts[i].xy_loc[1], char_pts[i+1].xy_loc[1]], 'w')

    # Plot right-running characteristic lines
    for i, vals in enumerate(rght_chars):
        for j, _ in enumerate(vals):
            if vals[j] < vals[-1]:
                plt.plot([char_pts[vals[j] - 1].xy_loc[0], char_pts[vals[j+1] - 1].xy_loc[0]],
                         [char_pts[vals[j] - 1].xy_loc[1], char_pts[vals[j+1] - 1].xy_loc[1]], 'w')

    # Plot lines that emanate from the throat to the first set of points
    for i in range(cn.N_LINES):
        plt.plot([0.0, char_pts[i].xy_loc[0]], [cn.RAD_THROAT, char_pts[i].xy_loc[1]], 'w')

    # Get x and y data from characteristic point locations
    x_data = [c.xy_loc[0] for c in char_pts]
    y_data = [c.xy_loc[1] for c in char_pts]

    # Form the x, y data into ranges for use with meshgrid
    x_range = np.linspace(min(x_data), max(x_data), len(x_data))
    y_range = np.linspace(min(y_data), max(y_data), len(y_data))

    # Form a 2-D mesh that works with contour plots
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)

    # Mach number data needs to be interpolated into a 2-D grid. Originally, the data is just stored
    # in a 1-D grid, where each Mach number is associated with a point in space. This interpolates
    # the Mach number data into a 2-D domain that can be used with a contour plot
    z_mesh = griddata((x_data, y_data), [c.mach_num for c in char_pts], (x_mesh, y_mesh))

    # Mach number contours between characteristic lines (flip y coordinate to plot below the
    # characteristic lines for clarity)
    plt.contourf(x_mesh, -y_mesh, z_mesh, cmap='magma', levels=cn.N_LINES)

    # Show final design information
    plt.axis('equal')
    plt.title(f'Input: $M_\\mathrm{{e}}={cn.EXIT_MACH}$, $\\gamma={cn.GAMMA}$, \
                $N_\\mathrm{{lines}}={cn.N_LINES}$, $r_\\mathrm{{t}}={cn.RAD_THROAT}$ \n \
                Exit Mach Number: {char_pts[-1].mach_num} \n \
                Calculated $A/A^*={calcd_area_ratio}$, Ideal $A/A^*={ideal_area_ratio}$, \
                Error: {percent_error}\\%')
    plt.xlabel('Nozzle Length $x$, [m]')
    plt.ylabel('Nozzle Height $y$, [m]')
    plt.colorbar(label='Mach Number, $M$ [-]')
    plt.show()

def data(x_data: list[float], y_data: list[float]):
    '''
    Generates a .csv file containing the upper wall data for use in external programs.

    Args:
        x_data (list[float]): x coordinates for points on the wall
        y_data (list[float]): y coordinates for points on the wall
    '''

    points = pd.DataFrame(np.array([x_data, y_data]).T, columns=['x', 'y'])

    points.to_csv(cn.PATH, index=False)
