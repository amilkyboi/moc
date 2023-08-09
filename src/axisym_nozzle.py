# module axisym_nozzle
'''
Axisymmetric supersonic nozzle design using the 2-D method of characteristics.
'''

import numpy as np

def testing():
    '''
    Testing.
    '''

    # given a throat radius
    t_rad = 1.0

    # place the initial flow angle as some small number
    flow_ang = np.deg2rad(0.375)
    # set pm angle as the same
    pran_ang = flow_ang

    k_neg = flow_ang + pran_ang
    k_pos = flow_ang - pran_ang

    # initial Mach is one on the sonic line
    mach_num = 1.0

    # set change in radius as some small number
    d_rad = 0.1

    # get next Riemann invariants from finite differencing
    k_neg = k_neg + (d_rad / t_rad) * (1 / (np.sqrt(mach_num**2 - 1) - (1 / np.tan(flow_ang))))
    k_pos = k_pos - (d_rad / t_rad) * (1 / (np.sqrt(mach_num**2 - 1) + (1 / np.tan(flow_ang))))

    flow_ang = 0.5 * (k_neg + k_pos)
    pran_ang = 0.5 * (k_neg - k_pos)

    print(np.rad2deg(flow_ang))
    print(np.rad2deg(pran_ang))


def main():
    '''
    Runs the program.
    '''

    testing()

if __name__ == '__main__':
    main()
