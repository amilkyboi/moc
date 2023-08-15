# module main
'''
Does what it says on the tin.
'''

import numpy as np

import initializer as init
import constants as cn
import output as out
import nozzle
import fan

def main():
    '''
    Runs the program.
    '''

    n_noz_pts = init.num_noz_pts()

    # Initialize the characteristic point with basic known values prior to performing MOC
    char_noz_pts = init.init_noz_pts(n_noz_pts)

    # Perform MOC and mutate the characteristic points accordingly
    char_noz_pts = nozzle.method_of_characteristics(char_noz_pts, n_noz_pts)

    # Since point (a), the point at the sharp throat of the nozzle, is not actually a characteristic
    # point, it needs to be added to the wall points manually

    # This is easy since we know the nozzle design is centered at the origin x-wise and begins at
    # the sharp throat
    x_wall_noz = [0.0]
    y_wall_noz = [cn.RAD_THROAT]

    # Add the positions of the points calculated using MOC separately based on if they fall upon the
    # wall or not
    x_char_noz = []
    y_char_noz = []
    for i in range(0, n_noz_pts):
        if not char_noz_pts[i].on_wall:
            x_char_noz.append(char_noz_pts[i].xy_loc[0])
            y_char_noz.append(char_noz_pts[i].xy_loc[1])

        if char_noz_pts[i].on_wall:
            x_wall_noz.append(char_noz_pts[i].xy_loc[0])
            y_wall_noz.append(char_noz_pts[i].xy_loc[1])

    n_fan_pts    = init.num_fan_pts()
    char_fan_pts = init.init_fan_pts(n_fan_pts)
    char_fan_pts = fan.method_of_characteristics(char_fan_pts, n_fan_pts, y_wall_noz[-1])

    # Expansion fan points
    x_char_fan = []
    y_char_fan = []
    for i in range(0, n_fan_pts):
        x_char_fan.append(char_fan_pts[i].xy_loc[0])
        y_char_fan.append(char_fan_pts[i].xy_loc[1])

    # Area ratio of the final nozzle design, A/A*

    # Since this nozzle design is two-dimensional, the ratio between the height of the last
    # wall point and the nozzle throat radius can be used as the area ratio
    calcd_area_ratio = char_noz_pts[-1].xy_loc[1] / cn.RAD_THROAT

    # Ideal area ratio using isentropic relations
    ideal_area_ratio = (0.5 * (cn.GAMMA + 1))**(-(cn.GAMMA + 1) / (2 * (cn.GAMMA - 1))) * (1/cn.EXIT_MACH) * \
                       (1 + 0.5 * (cn.GAMMA - 1) * cn.EXIT_MACH**2)**((cn.GAMMA + 1) / (2 * (cn.GAMMA - 1)))

    # Percent difference in area ratios
    percent_error = 100 * np.abs(ideal_area_ratio - calcd_area_ratio) / \
                   (0.5 * (ideal_area_ratio + calcd_area_ratio))

    if cn.INFO:
        print('OUTPUT:\n')
        print(f'Exit Mach Number: {char_noz_pts[-1].mach_num}')
        print(f'Ideal A/A*: {ideal_area_ratio}')
        print(f'Calculated A/A*: {calcd_area_ratio}')
        print(f'Percent Error: {percent_error}')

    if cn.SAVE:
        out.data(x_wall_noz, y_wall_noz)

    if cn.PLOT_NOZ:
        out.plot((x_wall_noz, y_wall_noz), (x_char_noz, y_char_noz), calcd_area_ratio, \
                 ideal_area_ratio, percent_error, char_noz_pts, (x_char_fan, y_char_fan), char_fan_pts)

if __name__ == '__main__':
    main()
