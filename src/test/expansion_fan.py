# module expansion_fan
'''
Temporary file to incorporate expansion waves.
'''

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import numpy as np

from flow import newton_raphson

# Global design parameters
GAMMA:      float = 1.4
EXIT_MACH:  float = 2.4
RAD_THROAT: float = 1.0
N_LINES:    int   = 25
EXIT_PRES: float = 200
BACK_PRES: float = 100

# Global method switch for the inverse Prandtl-Meyer function
METHOD: str = 'newton'

# Display design information to the terminal
INFO: bool = False

# Save upper wall data to .csv
SAVE: bool = False
PATH: str  = '../data/example.csv'

# Plot nozzle contour
PLOT: bool = True
RAD_EXIT = 1

def number_of_points() -> int:
    '''
    Series expansion for the total number of characteristic points needed based on the selected
    number of characteristic lines

    Returns:
        int: number of characteristic points
    '''

    return int(N_LINES * 0.5 * (N_LINES + 1))

def initialize_points(n_points: int) -> list['CharPoint']:
    '''
    Initializes all points and adds an internal index attribute. Also finds and marks the points
    that lie on the wall and those that lie on the centerline.

    Args:
        n_points (int): number of characteristic points

    Returns:
        list['CharPoint']: list of characteristic point objects
    '''

    # Array for storing the list of points, note that each point is an object
    char_pts = []

    j = 1
    k = N_LINES

    # Since the indexing in literature begins at 1 instead of zero, the internal idx attribute of
    # each point will reflect this, hence why this loop begins at 1 instead of 0
    for i in range(1, n_points + 1):
        # Create an object for each point and set the index accordingly
        point = CharPoint(idx=i)

        # Mark centerline points
        if i == j:
            point.on_cent = True

            j += k
            k -= 1

        # Add each point object to the array
        char_pts.append(point)

    return char_pts

def mach_angle(mach_num: float) -> float:
    '''
    Calculates the Mach angle of the flow based on the Mach number.

    Args:
        mach_num (float): Mach number

    Returns:
        float: Mach angle in [rad]
    '''

    return np.arcsin(1 / mach_num)

def prandtl_meyer(mach_num: float) -> float:
    '''
    Calculates the Prandtl-Meyer angle using the Prandtl-Meyer function.

    Args:
        mach_num (float): Mach number

    Returns:
        float: Prandtl-Meyer angle in [rad]
    '''

    return np.sqrt((GAMMA + 1)/(GAMMA - 1)) * \
           np.arctan(np.sqrt((mach_num**2 - 1) * (GAMMA - 1)/(GAMMA + 1))) - \
           np.arctan(np.sqrt(mach_num**2 - 1))

def inverse_prandtl_meyer(pran_ang: float) -> float:
    '''
    Calculates the Mach number of the flow based on the Prandtl-Meyer angle using numerical
    inversion. Option 'newton' uses the Newton-Raphson method, option 'composite' uses an
    approximate method based on Taylor series expansions.

    Args:
        pran_ang (float): Prandtl-Meyer angle in [rad]

    Returns:
        float: Mach number
    '''

    # Approximate method adapted from "Inversion of the Prandtl-Meyer Relation," by I. M. Hall,
    # published Sept. 1975. Gives M with an error of less than 0.05% over the whole range with an
    # uncertainty in nu of less than 0.015 degrees.
    if METHOD == 'composite':
        lmb     = np.sqrt((GAMMA - 1)/(GAMMA + 1))
        k_0     = 4/(3*np.pi)*(1 + 1/lmb)
        eta_inf = (3*np.pi/(2*lmb + 2*lmb**2))**(2/3)
        a_1     = 0.5*eta_inf
        a_2     = (3 + 8*lmb**2)/40*eta_inf**2
        a_3     = (-1 + 328*lmb**2 + 104*lmb**4)/2800*eta_inf**3
        d_1     = a_1 - 1 - (a_3 - k_0)/(a_2 - k_0)
        d_2     = a_2 - a_1 - ((a_1 - 1)*(a_3 - k_0))/(a_2 - k_0)
        d_3     = ((a_3 - k_0)*(a_1 - k_0))/(a_2 - k_0) - a_2 + k_0
        e_1     = -1 - (a_3 - k_0)/(a_2 - k_0)
        e_2     = -1 - e_1
        nu_inf  = 0.5*np.pi*(1/lmb-1)
        y_0     = (pran_ang/nu_inf)**(2/3)

        mach_num = (1 + d_1*y_0 + d_2*y_0**2 + d_3*y_0**3)/(1 + e_1*y_0 + e_2*y_0**2)

        return mach_num

    elif METHOD == 'newton':
        # Traditional root finding technique using the Newton-Raphson method
        def func(mach_num):
            return np.sqrt((GAMMA + 1)/(GAMMA - 1)) * \
                   np.arctan(np.sqrt((mach_num**2 - 1) * (GAMMA - 1)/(GAMMA + 1))) - \
                   np.arctan(np.sqrt(mach_num**2 - 1)) - pran_ang

        def dfunc(mach_num):
            return (np.sqrt(mach_num**2 - 1))/(mach_num + (GAMMA - 1)/2*mach_num**3)

        # Initial guess for Mach number
        # TODO: For a faster algorithm, the Mach number of the closest previous centerline point
        #       could be used instead, but guessing 2 for each iteration works well enough
        mach_0  = 2

        mach_num, _ = newton_raphson(func, dfunc, mach_0)

        return mach_num

    else:
        raise ValueError('Please enter either newton or composite')

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

def angle_divs(angle: float):
    '''
    Given the maximum expansion angle of the wall downstream of the throat, splits the angle into an
    n_divs number of equally spaced divisions.

    Args:
        angle (float): maximum wall angle in [rad]

    Returns:
        list[float]: list of equally spaced divisions in [rad]
    '''

    # Find the necessary change in angle for each step
    d_angle = angle / (N_LINES - 1)

    # Creates a list of angle divisions that begins at zero and ends at the input angle
    angles = []
    for i in range(N_LINES):
        angles.append(d_angle * i)

    return angles

def pressure_ratio(gamma, mach):
    '''
    Calculates the total (stagnation) pressure ratio, p/p0.
    '''
    return (1 + (gamma - 1)/2 * mach**2)**(-gamma/(gamma - 1))

def mach_from_pres(gamma, pres_ratio):
    '''
    blah.
    '''

    return np.sqrt((2 / (gamma - 1)) * (pres_ratio**(-(gamma - 1) / gamma) - 1))

def method_of_characteristics(char_pts: list['CharPoint'], n_points: int) -> list['CharPoint']:
    '''
    Performs the method of characteristics for a purely 2-D minimum-length supersonic nozzle.

    Args:
        char_pts (list['CharPoint']): list of characteristic point objects
        n_points (int): number of characteristic points

    Returns:
        list[float]: list of equally spaced divisions
    '''

    exit_pressure_ratio = pressure_ratio(GAMMA, EXIT_MACH)

    back_pressure_ratio = BACK_PRES / EXIT_PRES * exit_pressure_ratio

    back_mach = mach_from_pres(GAMMA, back_pressure_ratio)

    nu_3 = prandtl_meyer(back_mach)
    nu_1 = prandtl_meyer(EXIT_MACH)

    theta_3 = nu_3 - nu_1

    flow_ang_divs = angle_divs(theta_3)

    # Point (a)
    x_a = 0.0

    # Note the flow angle for the first point needs to be the same as the PM angle so that the K+
    # Riemann invariant is constant for the first set of characteristic points

    # We set the flow angle at the first point to zero because it is on the centerline
    # (This is already enforced from point initialization, but it is reiterated here for clarity)

    # The Prandtl-Meyer angle doesn't matter because we choose a starting Mach number as our design
    # initializer instead; we just choose 0 to match the flow angle and enforce the Riemann
    # invariant

    # A value close to 1 but not too close to cause issues with the algorithm
    # is valid, something in the range of 1.01 yields good results
    char_pts[0].flow_ang = 0.0
    char_pts[0].pran_ang = prandtl_meyer(EXIT_MACH)
    char_pts[0].mach_num = EXIT_MACH
    char_pts[0].mach_ang = mach_angle(char_pts[0].mach_num)

    # The slope of the characteristic line coming in to point 1 relative to the centerline is the
    # Mach angle minus the flow angle

    # Using x = y / tan(angle) the position of the first point can be found
    char_pts[0].xy_loc = [RAD_EXIT / (np.tan(char_pts[0].mach_ang - char_pts[0].flow_ang)), 0.0]

    # Keep track of Riemann invariants
    char_pts[0].k_neg = char_pts[0].flow_ang + char_pts[0].pran_ang
    char_pts[0].k_pos = char_pts[0].flow_ang - char_pts[0].pran_ang

    for i in range(1, N_LINES):
        # Previous point
        prv_pt = i - 1

        # The flow angle of point 1 is zero, so all subsequent points simply use the flow angle
        # divisions starting from index [1]
        char_pts[i].flow_ang = flow_ang_divs[i]
        char_pts[i].pran_ang = flow_ang_divs[i] + char_pts[0].pran_ang
        char_pts[i].mach_num = inverse_prandtl_meyer(char_pts[i].pran_ang)
        char_pts[i].mach_ang = mach_angle(char_pts[i].mach_num)

        char_pts[i].k_neg = char_pts[i].flow_ang + char_pts[i].pran_ang
        char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

        # In general, the slopes of the characteristic lines are approximated by:
        # slope(C-) = 0.5 * ((theta_1 - mu_1) + (theta_3 - mu_3))
        # slope(C+) = 0.5 * ((theta_2 + mu_2) + (theta_3 + mu_3))

        # Simply the angle of the characteristic line that eminates from the corner of the sharp
        # throat
        c_neg = char_pts[i].flow_ang - char_pts[i].mach_ang
        # Averaging the slope of the C+ characteristic lines from the previous point and the
        # current point
        c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                       char_pts[i].flow_ang + char_pts[i].mach_ang)

        char_pts[i].xy_loc = find_xy([x_a, RAD_EXIT], char_pts[prv_pt].xy_loc, c_neg, c_pos)

    # Remaining points (everything not on the first C+, C- characteristic line pair)
    j = 0
    k = N_LINES
    for i in range(N_LINES, n_points):
        # Previous point
        prv_pt = i - 1
        # Previous point that lies on the centerline (only used for centerline point calculations)
        cnt_pt = j

        if char_pts[i].on_cent:
            j += k
            k -= 1

            # For centerline points, we know the K- Riemann invariant is the same as the previous
            # upper point
            char_pts[i].k_neg = char_pts[i - k].k_neg

            # We also know that, since they fall on the centerline, their flow angle is
            # definitionally zero
            char_pts[i].flow_ang = 0.0

            # Since K- = flow_ang + pran_ang, the Prandtl-Meyer angle is easily found
            char_pts[i].pran_ang = char_pts[i].k_neg - char_pts[i].flow_ang

            # The rest of the flow parameters follow as standard
            char_pts[i].mach_num = inverse_prandtl_meyer(char_pts[i].pran_ang)
            char_pts[i].mach_ang = mach_angle(char_pts[i].mach_num)

            char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

            # Averaging the previous C- characteristic with the current one
            c_neg = 0.5 * (char_pts[i - k].flow_ang - char_pts[i - k].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            # The lower characteristic line coming into centerline points is from another centerline
            # point, which means the slope is zero
            c_pos = 0.0

            char_pts[i].xy_loc = find_xy(char_pts[i - k].xy_loc,
                                         char_pts[cnt_pt].xy_loc, c_neg, c_pos)

        if not char_pts[i].on_cent:
            # Internal flowfield points can be entirely characterized by the two characteristic
            # lines (C+ and C-) that pass through them

            # By definition, the K- and K+ constants hold from the previous top and bottom points,
            # respectively
            char_pts[i].k_neg = char_pts[i - k].k_neg
            char_pts[i].k_pos = char_pts[prv_pt].k_pos

            # Using the definition of the Riemann invariants, the flow angle and Mach angle can be
            # found
            char_pts[i].flow_ang = 0.5 * (char_pts[i].k_neg + char_pts[i].k_pos)
            char_pts[i].pran_ang = 0.5 * (char_pts[i].k_neg - char_pts[i].k_pos)

            # Other parameters follow
            char_pts[i].mach_num = inverse_prandtl_meyer(char_pts[i].pran_ang)
            char_pts[i].mach_ang = mach_angle(char_pts[i].mach_num)

            # Simple averaging to find the slope of the characteristic lines passing through
            c_neg = 0.5 * (char_pts[i - k].flow_ang - char_pts[i - k].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = find_xy(char_pts[i - k].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

    return char_pts

def find_rght_chars(num: int) -> list[int]:
    '''
    Finds the indices of the points that lie on each right-running characteristic by finding the
    triangular sequence of the input number.

    Args:
        num (int): characteristic point index

    Returns:
        list[int]: triangular sequence corresponding to the index of the input point
    '''

    sequence  = []
    start     = num
    increment = N_LINES - 1

    for _ in range(num):
        sequence.append(start)
        start += increment
        increment -= 1

    return sequence

def find_left_chars(num: int) -> list[list[int]]:
    '''
    Finds the indices of the points that lie on each left-running characteristic.

    Args:
        num (int): characteristic point index

    Returns:
        list[list[int]]: list for each point index containing which points follow in a decreasing
                         sequence
    '''

    result = []
    start  = 1

    # Decrement since the first list element is largest
    for i in range(num, 1, -1):
        sublist = list(range(start, start + i))
        result.append(sublist)
        start += i

    return result

def plot_expansion(char_data: tuple[list[float], list[float]], char_pts):
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

    # Interior and centerline points
    plt.scatter(char_data[0], char_data[1], facecolors='none', edgecolors='w')

    # Store the indices of the points that lie on the characteristics
    rght_chars = [find_rght_chars(num) for num in range(2, N_LINES + 1)]
    left_chars = find_left_chars(N_LINES)

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
    for i in range(N_LINES):
        plt.plot([0.0, char_pts[i].xy_loc[0]], [1, char_pts[i].xy_loc[1]], 'w')

    plt.show()

@dataclass
class CharPoint:
    '''
    Stores each characteristic point as a separate object in order to more easily assign flow
    parameters and access positional data.
    '''

    # Index
    idx: int

    # Wall and centerline parameters
    on_cent: bool = False

    # Flow angle, Prandtl-Meyer angle, and Mach angle
    flow_ang: float = 0
    pran_ang: float = 0
    mach_ang: float = 0

    # Riemann invariants
    k_neg: float = 0
    k_pos: float = 0

    # Position

    # This is some weird syntax, but dataclasses do not support mutable lists normally, so this has
    # to be used instead
    xy_loc: list[float] = field(default_factory=lambda: [0, 0])

    # Mach number
    mach_num: float = 0

def main():
    '''
    Runs the program.
    '''

    n_points = number_of_points()

    # Initialize the characteristic point with basic known values prior to performing MOC
    char_pts = initialize_points(n_points)

    # Perform MOC and mutate the characteristic points accordingly
    char_pts = method_of_characteristics(char_pts, n_points)

    # Since point (a), the point at the sharp throat of the nozzle, is not actually a characteristic
    # point, it needs to be added to the wall points manually

    # Add the positions of the points calculated using MOC separately based on if they fall upon the
    # wall or not
    x_char = []
    y_char = []
    for i in range(0, n_points):
        x_char.append(char_pts[i].xy_loc[0])
        y_char.append(char_pts[i].xy_loc[1])

    if PLOT:
        plot_expansion((x_char, y_char), char_pts)

if __name__ == '__main__':
    main()
