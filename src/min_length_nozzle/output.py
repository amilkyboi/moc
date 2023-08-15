# module output
'''
Manages output of the optionally generated figure and/or data file.
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
         char_fan_data: tuple[list[float], list[float]],
         char_noz_pts: list['CharPoint'],
         char_fan_pts: list['CharPoint']):
    '''
    Plotting interface for a visual representation of the nozzle geometry.

    Args:
        wall_noz_data (tuple[list[float], list[float]]): (x, y) location data for the wall points
        char_noz_data (tuple[list[float], list[float]]): (x, y) location data for the inner points
        char_fan_data (tuple[list[float], list[float]]): (x, y) location data for the fan
        char_noz_pts (list['CharPoint']): list of characteristic points for the nozzle
        char_fan_pts (list['CharPoint']): list of characteristic points for the fan
    '''

    # Area ratio of the final nozzle design, A/A*

    # Since this nozzle design is two-dimensional, the ratio between the height of the last
    # wall point and the nozzle throat radius can be used as the area ratio
    calcd_area_ratio = char_noz_pts[-1].xy_loc[1] / cn.RAD_THROAT

    # Ideal area ratio using isentropic relations
    ideal_area_ratio = (0.5 * (cn.GAMMA + 1))**(-(cn.GAMMA + 1) / (2 * (cn.GAMMA - 1))) * \
                       (1/cn.EXIT_MACH) * (1 + 0.5 * (cn.GAMMA - 1) * cn.EXIT_MACH**2) \
                        **((cn.GAMMA + 1) / (2 * (cn.GAMMA - 1)))

    # Percent difference in area ratios
    percent_error = 100 * np.abs(ideal_area_ratio - calcd_area_ratio) / \
                   (0.5 * (ideal_area_ratio + calcd_area_ratio))

    # Use the scienceplots module and a dark theme
    plt.style.use(['science', 'grid', 'dark_background'])

    mydpi = 96

    plt.figure(figsize=(cn.RES[0]/mydpi, cn.RES[1]/mydpi), dpi=mydpi)

    # NOZZLE ---------------------------------------------------------------------------------------

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
                plt.plot([char_noz_pts[vals[j] - 1].xy_loc[0],
                          char_noz_pts[vals[j+1] - 1].xy_loc[0]],
                         [char_noz_pts[vals[j] - 1].xy_loc[1],
                          char_noz_pts[vals[j+1] - 1].xy_loc[1]], 'w')

    # Plot lines that emanate from the throat to the first set of points
    for i in range(cn.N_LINES):
        plt.plot([0.0, char_noz_pts[i].xy_loc[0]], [cn.RAD_THROAT, char_noz_pts[i].xy_loc[1]], 'w')

    # EXPANSION FAN --------------------------------------------------------------------------------

    # Interior and centerline points (manually adding offset since the algorithm assumes an x-origin
    # of 0 instead of the end of the nozzle)
    plt.scatter([c + wall_noz_data[0][-1] for c in char_fan_data[0]],
                char_fan_data[1], facecolors='none', edgecolors='lightblue')

    # Store the indices of the points that lie on the characteristics
    rght_chars = [lines.find_rght_chars(num, False) for num in range(2, cn.N_LINES + 1)]
    left_chars = lines.find_left_chars(cn.N_LINES)

    # Plot left-running characteristic lines
    for i in range(len(char_fan_pts) - 1):
        for j, _ in enumerate(left_chars):
            # Ensure that the separate lines are not connected
            if char_fan_pts[i].idx in left_chars[j] and char_fan_pts[i+1].idx in left_chars[j]:
                plt.plot([char_fan_pts[i].xy_loc[0] + wall_noz_data[0][-1],
                          char_fan_pts[i+1].xy_loc[0] + wall_noz_data[0][-1]],
                         [char_fan_pts[i].xy_loc[1],
                          char_fan_pts[i+1].xy_loc[1]], 'lightblue')

    # Plot right-running characteristic lines
    for i, vals in enumerate(rght_chars):
        for j, _ in enumerate(vals):
            if vals[j] < vals[-1]:
                plt.plot([char_fan_pts[vals[j] - 1].xy_loc[0] + wall_noz_data[0][-1],
                          char_fan_pts[vals[j+1] - 1].xy_loc[0] + wall_noz_data[0][-1]],
                         [char_fan_pts[vals[j] - 1].xy_loc[1],
                          char_fan_pts[vals[j+1] - 1].xy_loc[1]], 'lightblue')

    # Plot lines that emanate from the throat to the first set of points
    for i in range(cn.N_LINES):
        plt.plot([wall_noz_data[0][-1], char_fan_pts[i].xy_loc[0] + wall_noz_data[0][-1]],
                 [wall_noz_data[1][-1], char_fan_pts[i].xy_loc[1]], 'lightblue')

    # MACH ISOLINES --------------------------------------------------------------------------------

    # Combine datasets for Mach contours throughout the entire domain
    char_pts = char_noz_pts + char_fan_pts

    # Again remembering to manually add the x-offset
    x_data = [c.xy_loc[0] for c in char_noz_pts] + \
             [c.xy_loc[0] + wall_noz_data[0][-1] for c in char_fan_pts]
    y_data = [c.xy_loc[1] for c in char_pts]

    x_range = np.linspace(min(x_data), max(x_data), len(x_data))
    y_range = np.linspace(min(y_data), max(y_data), len(y_data))

    # Generate a 2-D mesh throughout the domain
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)

    # Interpolation for the locations that lie between the actual characteristic points, otherwise
    # each (x, y) mesh location would not have an assigned Mach number
    z_mesh = griddata((x_data, y_data), [c.mach_num for c in char_pts], (x_mesh, y_mesh))

    plt.contourf(x_mesh, -y_mesh, z_mesh, cmap='magma', levels=cn.N_LINES)

    # Show final design information
    plt.axis('equal')
    plt.title(f'Input: $M_\\mathrm{{e}}={cn.EXIT_MACH}$, $\\gamma={cn.GAMMA}$, \
                $N_\\mathrm{{lines}}={cn.N_LINES}$, $r_\\mathrm{{t}}={cn.RAD_THROAT}$ \n \
                Exit Mach Number: {char_noz_pts[-1].mach_num}, \
                Fan Mach Number: {char_fan_pts[-1].mach_num}, Exit Pressure: {cn.EXIT_PRES} kPa, \
                Back Pressure: {cn.BACK_PRES} kPa \n \
                Calculated $A/A^*={calcd_area_ratio}$, Ideal $A/A^*={ideal_area_ratio}$, \
                Error: {percent_error}\\%')
    plt.xlabel('Nozzle Length $x$, [m]')
    plt.ylabel('Nozzle Height $y$, [m]')
    plt.colorbar(label='Mach Number, $M$ [-]')
    # plt.savefig(cn.IMG_PATH, dpi=mydpi * 2)
    plt.show()

def data(x_data: list[float], y_data: list[float]):
    '''
    Generates a .csv file containing the upper wall data for use in external programs.

    Args:
        x_data (list[float]): x coordinates for points on the upper wall
        y_data (list[float]): y coordinates for points on the upper wall
    '''

    points = pd.DataFrame(np.array([x_data, y_data]).T, columns=['x', 'y'])

    points.to_csv(cn.DATA_PATH, index=False)
