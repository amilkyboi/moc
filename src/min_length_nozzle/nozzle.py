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

import numpy as np

from initializer import CharPoint
import geometry as geom
import constants as cn
import flow

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
    max_wall_ang = 0.5 * flow.prandtl_meyer(cn.GAMMA, cn.EXIT_MACH)

    # Get the list of angle divisions and the division size
    flow_ang_divs = geom.angle_divs(max_wall_ang)

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
    char_pts[0].mach_ang = flow.mach_angle(char_pts[0].mach_num)

    # The slope of the characteristic line coming in to point 1 relative to the centerline is the
    # Mach angle minus the flow angle

    # Using x = y / tan(angle) the position of the first point can be found
    char_pts[0].xy_loc = [cn.RAD_THROAT / (np.tan(char_pts[0].mach_ang - char_pts[0].flow_ang)), 0]

    # Keep track of Riemann invariants
    char_pts[0].k_neg = char_pts[0].flow_ang + char_pts[0].pran_ang
    char_pts[0].k_pos = char_pts[0].flow_ang - char_pts[0].pran_ang

    # Point (2) through point (cn.N_LINES + 1) (a.k.a. the first wall point)
    for i in range(1, cn.N_LINES + 1):
        # Previous point
        prv_pt = i - 1

        if not char_pts[i].on_wall:
            # Starting with the points directly following point 1 (which falls on the centerline)

            # The flow angle of point 1 is zero, so all subsequent points simply use the flow angle
            # divisions starting from index [1]
            char_pts[i].flow_ang = flow_ang_divs[i]
            char_pts[i].pran_ang = flow_ang_divs[i]
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

            char_pts[i].xy_loc = geom.find_xy([x_a, cn.RAD_THROAT], char_pts[prv_pt].xy_loc, c_neg, c_pos)

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

            char_pts[i].xy_loc = geom.find_xy([x_a, cn.RAD_THROAT], char_pts[prv_pt].xy_loc, c_neg, c_pos)

    # Remaining points (everything not on the first C+, C- characteristic line pair)
    j = 0
    for i in range(cn.N_LINES + 1, n_points):
        # Previous point
        prv_pt = i - 1
        # Previous point vertically above the current point (y_prev > y_curr, x_prev < x_curr)
        top_pt = i - (cn.N_LINES - j)
        # Previous point that lies on the centerline (only used for centerline point calculations)
        cnt_pt = i - (cn.N_LINES - j) - 1

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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang, cn.METHOD)
            char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

            char_pts[i].k_pos = char_pts[i].flow_ang - char_pts[i].pran_ang

            # Averaging the previous C- characteristic with the current one
            c_neg = 0.5 * (char_pts[top_pt].flow_ang - char_pts[top_pt].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            # The lower characteristic line coming into centerline points is from another centerline
            # point, which means the slope is zero
            c_pos = 0.0

            char_pts[i].xy_loc = geom.find_xy(char_pts[top_pt].xy_loc,
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
            char_pts[i].mach_num = flow.inverse_prandtl_meyer(cn.GAMMA, char_pts[i].pran_ang, cn.METHOD)
            char_pts[i].mach_ang = flow.mach_angle(char_pts[i].mach_num)

            # Simple averaging to find the slope of the characteristic lines passing through
            c_neg = 0.5 * (char_pts[top_pt].flow_ang - char_pts[top_pt].mach_ang + \
                           char_pts[i].flow_ang - char_pts[i].mach_ang)
            c_pos = 0.5 * (char_pts[prv_pt].flow_ang + char_pts[prv_pt].mach_ang + \
                           char_pts[i].flow_ang + char_pts[i].mach_ang)

            char_pts[i].xy_loc = geom.find_xy(char_pts[top_pt].xy_loc,
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

            char_pts[i].xy_loc = geom.find_xy(char_pts[top_pt].xy_loc,
                                         char_pts[prv_pt].xy_loc, c_neg, c_pos)

            # Increment to note that a wall point has been passed
            j += 1

    return char_pts
