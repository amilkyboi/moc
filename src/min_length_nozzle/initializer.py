# module initializer
'''
Initializes the characteristic points into a list.
'''

from dataclasses import dataclass, field

import input as inp

def num_pts(is_nozzle: bool) -> int:
    '''
    Series expansion for the total number of characteristic points needed based on the selected
    number of characteristic lines.

    Args:
        is_nozzle (bool): True for nozzle, False for fan

    Returns:
        int: number of characteristic points
    '''

    if is_nozzle:
        return int(inp.N_LINES + inp.N_LINES * 0.5 * (inp.N_LINES + 1))

    return int(inp.N_LINES * 0.5 * (inp.N_LINES + 1))

def init_noz_pts(n_points: int) -> list['CharPoint']:
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
    # 1 + cn.N_LINES
    j = 1 + inp.N_LINES

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
            j += inp.N_LINES - k

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

def init_fan_pts(n_points: int) -> list['CharPoint']:
    '''
    Initializes all points and adds an internal index attribute. Also finds and marks the points
    that lie on the centerline.

    Args:
        n_points (int): number of characteristic points

    Returns:
        list['CharPoint']: list of characteristic point objects
    '''

    # Array for storing the list of points, note that each point is an object
    char_pts = []

    j = 1
    k = inp.N_LINES

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
