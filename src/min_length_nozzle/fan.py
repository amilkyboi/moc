# module expansion_fan
'''
Temporary file to incorporate expansion waves.
'''

import numpy as np

from initializer import CharPoint
import geometry as geom
import constants as cn
import flow

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

def method_of_characteristics(char_pts: list['CharPoint'], n_points: int, rad_exit: float) -> list['CharPoint']:
    '''
    Performs the method of characteristics for a purely 2-D minimum-length supersonic nozzle.

    Args:
        char_pts (list['CharPoint']): list of characteristic point objects
        n_points (int): number of characteristic points

    Returns:
        list[float]: list of equally spaced divisions
    '''

    exit_pressure_ratio = pressure_ratio(cn.GAMMA, cn.EXIT_MACH)

    back_pressure_ratio = cn.BACK_PRES / cn.EXIT_PRES * exit_pressure_ratio

    back_mach = mach_from_pres(cn.GAMMA, back_pressure_ratio)

    nu_3 = flow.prandtl_meyer(cn.GAMMA, back_mach)
    nu_1 = flow.prandtl_meyer(cn.GAMMA, cn.EXIT_MACH)

    theta_3 = nu_3 - nu_1

    flow_ang_divs = geom.angle_divs(theta_3)

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
    char_pts[0].pran_ang = flow.prandtl_meyer(cn.GAMMA, cn.EXIT_MACH)
    char_pts[0].mach_num = cn.EXIT_MACH
    char_pts[0].mach_ang = flow.mach_angle(char_pts[0].mach_num)

    # The slope of the characteristic line coming in to point 1 relative to the centerline is the
    # Mach angle minus the flow angle

    # Using x = y / tan(angle) the position of the first point can be found
    char_pts[0].xy_loc = [rad_exit / (np.tan(char_pts[0].mach_ang - char_pts[0].flow_ang)), 0.0]

    # Keep track of Riemann invariants
    char_pts[0].k_neg = char_pts[0].flow_ang + char_pts[0].pran_ang
    char_pts[0].k_pos = char_pts[0].flow_ang - char_pts[0].pran_ang

    for i in range(1, cn.N_LINES):
        # Previous point
        prv_pt = i - 1

        # The flow angle of point 1 is zero, so all subsequent points simply use the flow angle
        # divisions starting from index [1]
        char_pts[i].flow_ang = flow_ang_divs[i]
        char_pts[i].pran_ang = flow_ang_divs[i] + char_pts[0].pran_ang
        char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang, cn.METHOD)
        char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

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

        char_pts[i].xy_loc = geom.find_xy([x_a, rad_exit], char_pts[prv_pt].xy_loc, c_neg, c_pos)

    # Remaining points (everything not on the first C+, C- characteristic line pair)
    j = 0
    k = cn.N_LINES
    for i in range(cn.N_LINES, n_points):
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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang, cn.METHOD)
            char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

            char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

            # Averaging the previous C- characteristic with the current one
            c_neg = 0.5 * (char_pts[i - k].flow_ang - char_pts[i - k].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            # The lower characteristic line coming into centerline points is from another centerline
            # point, which means the slope is zero
            c_pos = 0.0

            char_pts[i].xy_loc = geom.find_xy(char_pts[i - k].xy_loc,
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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang, cn.METHOD)
            char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

            # Simple averaging to find the slope of the characteristic lines passing through
            c_neg = 0.5 * (char_pts[i - k].flow_ang - char_pts[i - k].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = geom.find_xy(char_pts[i - k].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

    return char_pts
