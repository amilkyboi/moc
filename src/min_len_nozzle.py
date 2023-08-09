# module min_len_nozzle
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

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

from newton_raphson import newton_raphson

# Global design parameters
GAMMA:      float = 1.4
MACH_E:     float = 2.4
RAD_THROAT: float = 1.0
N_LINES:    int   = 25

# Global method switch for the inverse Prandtl-Meyer function
METHOD: str = 'newton'

# Display design information to the terminal
INFO: bool = False

# Save upper wall data to .csv
SAVE: bool = False
PATH: str  = '../data/example.csv'

# Plot nozzle contour
PLOT: bool = True

def number_of_points() -> int:
    '''
    Series expansion for the total number of characteristic points needed based on the selected
    number of characteristic lines

    Returns:
        int: number of characteristic points
    '''

    return int(N_LINES + N_LINES * 0.5 * (N_LINES + 1))

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

    # The number of points that lie along the first C+ left-running characteristic line is equal to
    # 1 + N_LINES
    j = 1 + N_LINES

    # This is a counter that will increment by one each time a wall point is encountered, see the
    # loop below for details on numbering
    k = 0

    # Since the indexing in literature begins at 1 instead of zero, the internal idx attribute of
    # each point will reflect this, hence why this loop begins at 1 instead of 0
    for i in range(1, n_points + 1):
        # Create an object for each point and set the index accordingly
        point = CharPoint(idx=i)

        # First, j is the index of the first point that falls on the wall, so that point is marked
        # as a wall point (for 7 characteristic lines, point 8 is the first point on the wall)
        if i == j + k:
            point.on_wall = True

            # The j counter decreases by one each iteration because 1 characteristic point is 'lost'
            # for each C-, C+ pair that eminates from the throat of the nozzle

            # For 7 characteristic lines, the wall indices are: 8, 15, 21, 26, 30, 33, 35
            # Note that the change from one to the next decreases by one for each wall point
            # increment
            k += 1
            j += N_LINES - k

        # Add each point object to the array
        char_pts.append(point)

    # Again loop over everything to find centerline points. Here, the range begins at 0 since list
    # indexing is being performed and the internal idx attributes are not being changed / accessed
    for i in range(0, n_points):
        # The first point is placed on the centerline by definition, so its state is changed
        # accordingly
        if char_pts[i].idx == 1:
            char_pts[i].on_cent   = True
            char_pts[i].flow_ang  = 0
            char_pts[i].xy_loc[1] = 0

        # Since all wall points are essentially the 'end' of a C-, C+ characteristic line pair, the
        # first point on the next characteristic pair will always be a centerline point
        if i >= 1:
            if char_pts[i - 1].on_wall:
                char_pts[i].on_cent   = True
                char_pts[i].flow_ang  = 0
                char_pts[i].xy_loc[1] = 0

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

def method_of_characteristics(char_pts: list['CharPoint'], n_points: int) -> list['CharPoint']:
    '''
    Performs the method of characteristics for a purely 2-D minimum-length supersonic nozzle.

    Args:
        char_pts (list['CharPoint']): list of characteristic point objects
        n_points (int): number of characteristic points

    Returns:
        list[float]: list of equally spaced divisions
    '''

    # Find the maximum wall angle in radians
    max_wall_ang = 0.5 * prandtl_meyer(MACH_E)

    # Get the list of angle divisions and the division size
    flow_ang_divs = angle_divs(max_wall_ang)

    # Point (a)
    x_a = 0

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
    char_pts[0].pran_ang = 0.0
    char_pts[0].mach_num = 1.01
    char_pts[0].mach_ang = mach_angle(char_pts[0].mach_num)

    # The slope of the characteristic line coming in to point 1 relative to the centerline is the
    # Mach angle minus the flow angle

    # Using x = y / tan(angle) the position of the first point can be found
    char_pts[0].xy_loc = [RAD_THROAT / (np.tan(char_pts[0].mach_ang - char_pts[0].flow_ang)), 0]

    # Keep track of Riemann invariants
    char_pts[0].k_neg = char_pts[0].flow_ang + char_pts[0].pran_ang
    char_pts[0].k_pos = char_pts[0].flow_ang - char_pts[0].pran_ang

    # Point (2) through point (N_LINES + 1) (a.k.a. the first wall point)
    for i in range(1, N_LINES + 1):
        # Previous point
        prv_pt = i - 1

        if not char_pts[i].on_wall:
            # Starting with the points directly following point 1 (which falls on the centerline)

            # The flow angle of point 1 is zero, so all subsequent points simply use the flow angle
            # divisions starting from index [1]
            char_pts[i].flow_ang = flow_ang_divs[i]
            char_pts[i].pran_ang = flow_ang_divs[i]
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

            char_pts[i].xy_loc = find_xy([x_a, RAD_THROAT], char_pts[prv_pt].xy_loc, c_neg, c_pos)

        if char_pts[i].on_wall:
            # Only the first wall point falls within the outer loop parameters, so this loop only
            # covers one point

            # Wall points have the same flow parameters as the previous point by definition
            char_pts[i].flow_ang = char_pts[prv_pt].flow_ang
            char_pts[i].pran_ang = char_pts[prv_pt].pran_ang
            char_pts[i].mach_num = char_pts[prv_pt].mach_num
            char_pts[i].mach_ang = char_pts[prv_pt].mach_ang

            char_pts[i].k_neg = char_pts[i].flow_ang + char_pts[i].pran_ang
            char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

            # For the first wall point, the previous C- characteristic is just the max wall angle
            c_neg = max_wall_ang
            # Averaging the slope of the C+ lines for the previous and current points
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = find_xy([x_a, RAD_THROAT], char_pts[prv_pt].xy_loc, c_neg, c_pos)

    # Remaining points (everything not on the first C+, C- characteristic line pair)
    j = 0
    for i in range(N_LINES + 1, n_points):
        # Previous point
        prv_pt = i - 1
        # Previous point vertically above the current point (y_prev > y_curr, x_prev < x_curr)
        top_pt = i - (N_LINES - j)
        # Previous point that lies on the centerline (only used for centerline point calculations)
        cnt_pt = i - (N_LINES - j) - 1

        if char_pts[i].on_cent:
            # For centerline points, we know the K- Riemann invariant is the same as the previous
            # upper point
            char_pts[i].k_neg = char_pts[top_pt].k_neg

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
            c_neg = 0.5 * (char_pts[top_pt].flow_ang - char_pts[top_pt].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            # The lower characteristic line coming into centerline points is from another centerline
            # point, which means the slope is zero
            c_pos = 0.0

            char_pts[i].xy_loc = find_xy(char_pts[top_pt].xy_loc,
                                         char_pts[cnt_pt].xy_loc, c_neg, c_pos)

        if (not char_pts[i].on_cent) and (not char_pts[i].on_wall):
            # Internal flowfield points can be entirely characterized by the two characteristic
            # lines (C+ and C-) that pass through them

            # By definition, the K- and K+ constants hold from the previous top and bottom points,
            # respectively
            char_pts[i].k_neg = char_pts[top_pt].k_neg
            char_pts[i].k_pos = char_pts[prv_pt].k_pos

            # Using the definition of the Riemann invariants, the flow angle and Mach angle can be
            # found
            char_pts[i].flow_ang = 0.5 * (char_pts[i].k_neg + char_pts[i].k_pos)
            char_pts[i].pran_ang = 0.5 * (char_pts[i].k_neg - char_pts[i].k_pos)

            # Other parameters follow
            char_pts[i].mach_num = inverse_prandtl_meyer(char_pts[i].pran_ang)
            char_pts[i].mach_ang = mach_angle(char_pts[i].mach_num)

            # Simple averaging to find the slope of the characteristic lines passing through
            c_neg = 0.5 * (char_pts[top_pt].flow_ang - char_pts[top_pt].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = find_xy(char_pts[top_pt].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

        if char_pts[i].on_wall:
            # As before, points on the wall inheret the flow characteristics from the previous point
            char_pts[i].flow_ang = char_pts[prv_pt].flow_ang
            char_pts[i].pran_ang = char_pts[prv_pt].pran_ang
            char_pts[i].mach_num = char_pts[prv_pt].mach_num
            char_pts[i].mach_ang = char_pts[prv_pt].mach_ang

            char_pts[i].k_neg = char_pts[i].flow_ang + char_pts[i].pran_ang
            char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

            # For wall points, the C- characteristic is just the wall angle since there are no
            # points in the mesh that lie above the wall

            # To find the current angle of the wall, the flow angle of the previous and current
            # points are averaged (lines eminate before and after the point, so taking the average
            # is an easy way to get the slope "at" the point itself)
            c_neg = 0.5 * (char_pts[top_pt].flow_ang + char_pts[i].flow_ang)
            # The C+ line emanates from the previous point, so its calculations are done as normal
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = find_xy(char_pts[top_pt].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

            # Increment to note that a wall point has been passed
            j += 1

    return char_pts

def plotting(wall_data: tuple[list[float], list[float]],
             char_data: tuple[list[float], list[float]],
             calcd_area_ratio: float, ideal_area_ratio: float, percent_error: float):
    '''
    Plotting interface for a visual representation of the nozzle geometry.

    Args:
        wall_data (tuple[list[float], list[float]]): (x, y) coordinates for points on the wall
        char_data (tuple[list[float], list[float]]): (x, y) coordinates for all other points
        calcd_area_ratio (float): calculated area ratio using throat radius and final y coordinate
        ideal_area_ratio (float): ideal area ratio calculated using isentropic relations
        percent_error (float): % error between the calculated and ideal area ratios
    '''

    # Colors for internal points
    cols = np.sqrt(np.array(char_data[0])**2 + np.array(char_data[1])**2)
    norm = cols / cols.max()
    cmap = plt.cm.magma(norm)

    # Use the scienceplots module and a dark theme
    plt.style.use(['science', 'grid', 'dark_background'])

    # Lines connecting the top and bottom wall points
    plt.plot(wall_data[0], wall_data[1], 'w')
    plt.plot(wall_data[0], [-y for y in wall_data[1]], 'w')

    # Top and bottom wall points
    plt.scatter(wall_data[0], wall_data[1], facecolors='none', edgecolors='w')
    plt.scatter(wall_data[0], [-y for y in wall_data[1]], facecolors='none', edgecolors='w')

    # Interior and centerline points
    plt.scatter(char_data[0], char_data[1], facecolors='none', edgecolors=cmap)
    plt.scatter(char_data[0], [-y for y in char_data[1]], facecolors='none', edgecolors=cmap)

    # Other information
    plt.axis('equal')
    plt.title(f'Input: $M_\\mathrm{{e}}={MACH_E}$, $\\gamma={GAMMA}$, \
                $N_\\mathrm{{lines}}={N_LINES}$, $r_\\mathrm{{t}}={RAD_THROAT}$ \n \
                Calculated $A/A^*={calcd_area_ratio}$, Ideal $A/A^*={ideal_area_ratio}$, \
                Error: {percent_error}\\%')
    plt.xlabel('Nozzle Length $x$, [m]')
    plt.ylabel('Nozzle Height $y$, [m]')
    plt.show()

def data_output(x_data: list[float], y_data: list[float]):
    '''
    Generates a .csv file containing the upper wall data for use in external programs.

    Args:
        x_data (list[float]): x coordinates for points on the wall
        y_data (list[float]): y coordinates for points on the wall
    '''

    points = pd.DataFrame(np.array([x_data, y_data]).T, columns=['x', 'y'])

    points.to_csv(PATH, index=False)

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
    on_wall: bool = False

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

    # This is easy since we know the nozzle design is centered at the origin x-wise and begins at
    # the sharp throat
    x_wall = [0.0]
    y_wall = [RAD_THROAT]

    # Add the positions of the points calculated using MOC separately based on if they fall upon the
    # wall or not
    x_char = []
    y_char = []
    for i in range(0, n_points):
        if not char_pts[i].on_wall:
            x_char.append(char_pts[i].xy_loc[0])
            y_char.append(char_pts[i].xy_loc[1])

        if char_pts[i].on_wall:
            x_wall.append(char_pts[i].xy_loc[0])
            y_wall.append(char_pts[i].xy_loc[1])

    # Area ratio of the final nozzle design, A/A*

    # Since this nozzle design is two-dimensional, the ratio between the height of the last
    # wall point and the nozzle throat radius can be used as the area ratio
    calcd_area_ratio = char_pts[-1].xy_loc[1] / RAD_THROAT

    # Ideal area ratio using isentropic relations
    ideal_area_ratio = (0.5 * (GAMMA + 1))**(-(GAMMA + 1) / (2 * (GAMMA - 1))) * (1/MACH_E) * \
                       (1 + 0.5 * (GAMMA - 1) * MACH_E**2)**((GAMMA + 1) / (2 * (GAMMA - 1)))

    # Percent difference in area ratios
    percent_error = 100 * np.abs(ideal_area_ratio - calcd_area_ratio) / \
                   (0.5 * (ideal_area_ratio + calcd_area_ratio))

    if INFO:
        print('OUTPUT:\n')
        print(f'Ideal A/A*: {ideal_area_ratio}')
        print(f'Calculated A/A*: {calcd_area_ratio}')
        print(f'Percent Error: {percent_error}')

    if SAVE:
        data_output(x_wall, y_wall)

    if PLOT:
        plotting((x_wall, y_wall), (x_char, y_char), calcd_area_ratio, \
                 ideal_area_ratio, percent_error)

if __name__ == '__main__':
    main()
