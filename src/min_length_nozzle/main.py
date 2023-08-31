# module main
'''
Applies the method of characteristics for the design of a minimum-length supersonic nozzle. The
methods applied here assume that the flow inside the nozzle is:
- steady
- adiabatic
- two-dimensional
- irrotational
- shock-free
- isentropic
- supersonic

This code assumes a straight vertical sonic line at the nozzle throat and neglects the expansion
section entirely. Only the straightening section is considered, where the wall angle steadily
decreases.

Note that the compatibility equations used here are not valid for axisymmetric flow; this tool
should not be used for axisymmetric nozzle designs.

Requirements:
    Python 3.7+ (for dataclasses), matplotlib, scienceplots, pandas, numpy
'''

import initializer as init
import output as out
import input as inp
import nozzle
import fan

def main():
    '''
    Runs the program.
    '''

    # Get number of characteristic points that will make up the nozzle section
    n_noz_pts = init.num_pts(True)

    # Initialize the characteristic point with basic known values prior to performing MOC
    char_noz_pts = init.init_noz_pts(n_noz_pts)

    # Perform MOC and mutate the characteristic points accordingly
    char_noz_pts = nozzle.method_of_characteristics(char_noz_pts, n_noz_pts)

    # Since point (a), the point at the sharp throat of the nozzle, is not actually a characteristic
    # point, it needs to be added to the wall points manually

    # This is easy since we know the nozzle design is centered at the origin x-wise and begins at
    # the sharp throat
    x_wall_noz = [0.0]
    y_wall_noz = [inp.RAD_THROAT]

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

    # Same set of initialization and calculation steps for the expansion fan
    n_fan_pts    = init.num_pts(False)
    char_fan_pts = init.init_fan_pts(n_fan_pts)
    char_fan_pts = fan.method_of_characteristics(char_fan_pts, n_fan_pts, y_wall_noz[-1])

    # Expansion fan points
    x_char_fan = []
    y_char_fan = []
    for i in range(0, n_fan_pts):
        x_char_fan.append(char_fan_pts[i].xy_loc[0])
        y_char_fan.append(char_fan_pts[i].xy_loc[1])

    if inp.SAVE:
        out.data(x_wall_noz, y_wall_noz)

    if inp.PLOT:
        out.plot((x_wall_noz, y_wall_noz), (x_char_noz, y_char_noz), (x_char_fan, y_char_fan),
                 char_noz_pts, char_fan_pts)

if __name__ == '__main__':
    main()
