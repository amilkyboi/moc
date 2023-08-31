# module geometry
'''
Geometric calculations for angles and Cartesian coordinates.
'''

import numpy as np
import input as inp

def angle_divs(angle: float):
    '''
    Given a desired final angle, splits said angle into an equal number of divisions depending on
    the number of characteristic lines selected.

    Args:
        angle (float): desired final angle [rad]

    Returns:
        list[float]: list of equally spaced divisions in [rad]
    '''

    # Find the necessary change in angle for each step
    d_angle = angle / (inp.N_LINES - 1)

    # Creates a list of angle divisions that begins at zero and ends at the input angle
    angles = []
    for i in range(inp.N_LINES):
        angles.append(d_angle * i)

    return angles

def find_xy(xy_top: list[float], xy_bot: list[float],
            c_neg: float, c_pos: float) -> list[float]:
    '''
    Calculates the (x, y) position of a characteristic point. The required parameters are the (x, y)
    positions of the characteristic points directly upstream that fall along the C+ and C-
    characteristic lines. The slope of these two lines is also needed.

    Args:
        xy_top (list[float]): (x, y) coordinates of the previous point along the C- char. line
        xy_bot (list[float]): (x, y) coordinates of the previous point along the C+ char. line
        c_neg (float): slope of the previous C- line in [rad]
        c_pos (float): slope of the previous C+ line in [rad]

    Returns:
        list[float]: (x, y) coordinates of the current point
    '''

    # System of two eqations for two unknowns, which can be derived from:
    # (y3 - y1) / (x3 - x1) = tan(dy/dx of C-)
    # (y3 - y2) / (x3 - x2) = tan(dy/dx of C+)
    x_loc = (xy_top[0]*np.tan(c_neg) - xy_bot[0]*np.tan(c_pos) + xy_bot[1] - xy_top[1]) / \
            (np.tan(c_neg) - np.tan(c_pos))

    y_loc = (np.tan(c_neg)*np.tan(c_pos)*(xy_top[0] - xy_bot[0]) + np.tan(c_neg)*xy_bot[1] - \
             np.tan(c_pos)*xy_top[1])/(np.tan(c_neg) - np.tan(c_pos))

    return [x_loc, y_loc]
