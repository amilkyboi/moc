# module newton_raphson
'''
Evaluates a root of f(x) = 0 using the Newton-Raphson method.
'''

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
