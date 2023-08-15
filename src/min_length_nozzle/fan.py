# module expansion_fan
'''
Method of characteristics algorithm for the calculation of the expansion fan following the nozzle.
'''

import numpy as np

from initializer import CharPoint
import geometry as geom
import constants as cn
import flow

def method_of_characteristics(char_pts: list['CharPoint'], n_points: int,
                              rad_exit: float) -> list['CharPoint']:
    '''
    Performs the method of characteristics for a 2-D expansion fan following the exit of a nozzle
    into freestream conditions.

    Args:
        char_pts (list['CharPoint']): list of characteristic point objects
        n_points (int): number of characteristic points
        rad_exit (float): radius of the nozzle exit

    Returns:
        list['CharPoint']: list of characteristic points
    '''

    exit_pressure_ratio = flow.pressure_ratio(cn.GAMMA, cn.EXIT_MACH)

    back_pressure_ratio = cn.BACK_PRES / cn.EXIT_PRES * exit_pressure_ratio

    back_mach = flow.mach_from_pres(cn.GAMMA, back_pressure_ratio)

    nu_3 = flow.prandtl_meyer(cn.GAMMA, back_mach)
    nu_1 = flow.prandtl_meyer(cn.GAMMA, cn.EXIT_MACH)

    theta_3 = nu_3 - nu_1

    flow_ang_divs = geom.angle_divs(theta_3)

    # Point (a)
    x_a = 0.0

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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang,
                                                              cn.METHOD)
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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang,
                                                              cn.METHOD)
            char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

            # Simple averaging to find the slope of the characteristic lines passing through
            c_neg = 0.5 * (char_pts[i - k].flow_ang - char_pts[i - k].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = geom.find_xy(char_pts[i - k].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

    return char_pts
