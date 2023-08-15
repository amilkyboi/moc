# module flow
'''
Common flow functions.
'''

import numpy as np

def newton_raphson(func, dfunc, root: float, n_max: int=100, tol: float=1e-11) -> tuple[float, int]:
    '''
    Simple implementation of the Newton-Raphson method. Returns the final root guess and the number
    of iterations required.
    '''

    i = 0

    for i in range(n_max):
        change = func(root)/dfunc(root)
        root -= change

        if abs(change) < tol:
            break

    return root, i

def mach_angle(mach_num: float) -> float:
    '''
    Calculates the Mach angle of the flow based on the Mach number.

    Args:
        mach_num (float): Mach number

    Returns:
        float: Mach angle in [rad]
    '''

    return np.arcsin(1 / mach_num)

def prandtl_meyer(gamma: float, mach_num: float) -> float:
    '''
    Calculates the Prandtl-Meyer angle using the Prandtl-Meyer function.

    Args:
        gamma (float): ratio of specific heats
        mach_num (float): Mach number

    Returns:
        float: Prandtl-Meyer angle in [rad]
    '''

    return np.sqrt((gamma + 1)/(gamma - 1)) * \
           np.arctan(np.sqrt((mach_num**2 - 1) * (gamma - 1)/(gamma + 1))) - \
           np.arctan(np.sqrt(mach_num**2 - 1))

def inverse_prandtl_meyer(gamma: float, pran_ang: float, method: str) -> float:
    '''
    Calculates the Mach number of the flow based on the Prandtl-Meyer angle using numerical
    inversion. Option 'newton' uses the Newton-Raphson method, option 'composite' uses an
    approximate method based on Taylor series expansions.

    Args:
        gamma (float): ratio of specific heats
        pran_ang (float): Prandtl-Meyer angle in [rad]
        method: which method is used to calculate the inverse Prandtl-Meyer function, either newton
                or composite

    Returns:
        float: Mach number
    '''

    # Approximate method adapted from "Inversion of the Prandtl-Meyer Relation," by I. M. Hall,
    # published Sept. 1975. Gives M with an error of less than 0.05% over the whole range with an
    # uncertainty in nu of less than 0.015 degrees.
    if method == 'composite':
        lmb     = np.sqrt((gamma - 1)/(gamma + 1))
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

    elif method == 'newton':
        # Traditional root finding technique using the Newton-Raphson method
        def func(mach_num):
            return np.sqrt((gamma + 1)/(gamma - 1)) * \
                   np.arctan(np.sqrt((mach_num**2 - 1) * (gamma - 1)/(gamma + 1))) - \
                   np.arctan(np.sqrt(mach_num**2 - 1)) - pran_ang

        def dfunc(mach_num):
            return (np.sqrt(mach_num**2 - 1))/(mach_num + (gamma - 1)/2*mach_num**3)

        # Initial guess for Mach number
        # TODO: For a faster algorithm, the Mach number of the closest previous centerline point
        #       could be used instead, but guessing 2 for each iteration works well enough
        mach_0  = 2

        mach_num, _ = newton_raphson(func, dfunc, mach_0)

        return mach_num

    else:
        raise ValueError('Please enter either newton or composite')
