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

def plot(wall_noz_data: tuple[list[float], list[float]],
         char_noz_data: tuple[list[float], list[float]],
         calcd_area_ratio: float, ideal_area_ratio: float,
         percent_error: float, char_noz_pts: list['CharPoint'],
         char_fan_data, char_fan_pts: list['CharPoint']):
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
    plt.plot(wall_noz_data[0], wall_noz_data[1], 'lightgreen')
    plt.plot(wall_noz_data[0], [-y for y in wall_noz_data[1]], 'lightgreen')

    # Top wall points
    plt.scatter(wall_noz_data[0], wall_noz_data[1], facecolors='none', edgecolors='lightgreen')

    # Interior and centerline points
    plt.scatter(char_noz_data[0], char_noz_data[1], facecolors='none', edgecolors='w')

    # Store the indices of the points that lie on the characteristics
    rght_chars = [lines.find_rght_chars(num, True) for num in range(2, cn.N_LINES + 1)]
    left_chars =  lines.find_left_chars(cn.N_LINES + 1)

    # Plot left-running characteristic lines
    for i in range(len(char_noz_pts) - 1):
        for j, _ in enumerate(left_chars):
            # Ensure that the separate lines are not connected
            if char_noz_pts[i].idx in left_chars[j] and char_noz_pts[i+1].idx in left_chars[j]:
                plt.plot([char_noz_pts[i].xy_loc[0], char_noz_pts[i+1].xy_loc[0]],
                         [char_noz_pts[i].xy_loc[1], char_noz_pts[i+1].xy_loc[1]], 'w')

    # Plot right-running characteristic lines
    for i, vals in enumerate(rght_chars):
        for j, _ in enumerate(vals):
            if vals[j] < vals[-1]:
                plt.plot([char_noz_pts[vals[j] - 1].xy_loc[0], char_noz_pts[vals[j+1] - 1].xy_loc[0]],
                         [char_noz_pts[vals[j] - 1].xy_loc[1], char_noz_pts[vals[j+1] - 1].xy_loc[1]], 'w')

    # Plot lines that emanate from the throat to the first set of points
    for i in range(cn.N_LINES):
        plt.plot([0.0, char_noz_pts[i].xy_loc[0]], [cn.RAD_THROAT, char_noz_pts[i].xy_loc[1]], 'w')

    if cn.PLOT_FAN:
        # Interior and centerline points
        plt.scatter(char_fan_data[0] + wall_noz_data[0][-1], char_fan_data[1], facecolors='none', edgecolors='lightblue')

        # Store the indices of the points that lie on the characteristics
        rght_chars = [lines.find_rght_chars(num, False) for num in range(2, cn.N_LINES + 1)]
        left_chars = lines.find_left_chars(cn.N_LINES)

        # Plot left-running characteristic lines
        for i in range(len(char_fan_pts) - 1):
            for j, _ in enumerate(left_chars):
                # Ensure that the separate lines are not connected
                if char_fan_pts[i].idx in left_chars[j] and char_fan_pts[i+1].idx in left_chars[j]:
                    plt.plot([char_fan_pts[i].xy_loc[0] + wall_noz_data[0][-1], char_fan_pts[i+1].xy_loc[0] + wall_noz_data[0][-1]],
                             [char_fan_pts[i].xy_loc[1], char_fan_pts[i+1].xy_loc[1]], 'lightblue')

        # Plot right-running characteristic lines
        for i, vals in enumerate(rght_chars):
            for j, _ in enumerate(vals):
                if vals[j] < vals[-1]:
                    plt.plot([char_fan_pts[vals[j] - 1].xy_loc[0] + wall_noz_data[0][-1], char_fan_pts[vals[j+1] - 1].xy_loc[0] + wall_noz_data[0][-1]],
                            [char_fan_pts[vals[j] - 1].xy_loc[1], char_fan_pts[vals[j+1] - 1].xy_loc[1]], 'lightblue')

        # Plot lines that emanate from the throat to the first set of points
        for i in range(cn.N_LINES):
            plt.plot([wall_noz_data[0][-1], char_fan_pts[i].xy_loc[0] + wall_noz_data[0][-1]], [wall_noz_data[1][-1], char_fan_pts[i].xy_loc[1]], 'lightblue')

        char_pts = char_noz_pts + char_fan_pts

        x_data = [c.xy_loc[0] for c in char_noz_pts] + [c.xy_loc[0] + wall_noz_data[0][-1] for c in char_fan_pts]
        y_data = [c.xy_loc[1] for c in char_pts]

        x_range = np.linspace(min(x_data), max(x_data), len(x_data))
        y_range = np.linspace(min(y_data), max(y_data), len(y_data))

        x_mesh, y_mesh = np.meshgrid(x_range, y_range)

        z_mesh = griddata((x_data, y_data), [c.mach_num for c in char_pts], (x_mesh, y_mesh))

        plt.contourf(x_mesh, -y_mesh, z_mesh, cmap='magma', levels=cn.N_LINES)

    else:
        x_data = [c.xy_loc[0] for c in char_noz_pts]
        y_data = [c.xy_loc[1] for c in char_noz_pts]

        x_range = np.linspace(min(x_data), max(x_data), len(x_data))
        y_range = np.linspace(min(y_data), max(y_data), len(y_data))

        x_mesh, y_mesh = np.meshgrid(x_range, y_range)

        z_mesh = griddata((x_data, y_data), [c.mach_num for c in char_noz_pts], (x_mesh, y_mesh))

        plt.contourf(x_mesh, -y_mesh, z_mesh, cmap='magma', levels=cn.N_LINES)

    # Show final design information
    plt.axis('equal')
    plt.title(f'Input: $M_\\mathrm{{e}}={cn.EXIT_MACH}$, $\\gamma={cn.GAMMA}$, \
                $N_\\mathrm{{lines}}={cn.N_LINES}$, $r_\\mathrm{{t}}={cn.RAD_THROAT}$ \n \
                Exit Mach Number: {char_noz_pts[-1].mach_num}, Fan Mach Number: {char_fan_pts[-1].mach_num}, Exit Pressure: {cn.EXIT_PRES} kPa, Back Pressure: {cn.BACK_PRES} kPa \n \
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
