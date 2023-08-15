# module lines
'''
Characteristic lines.
'''

import constants as cn

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
    increment = cn.N_LINES

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
