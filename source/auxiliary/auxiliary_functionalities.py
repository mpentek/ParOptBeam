from math import ceil, log10
import sys
import numpy as np
# TODO: clean up these function, see how to make the shear beam / additional rotational stiffness

CUST_MAGNITUDE = 4


def magnitude(x):
    # NOTE: ceil is supposed to be correct for positive values
    return int(ceil(log10(x)))


def map_lin_to_log(val, base=10**3):
    # it has to be defined with a min=0.0 and max=1.0
    # TODO: implment check
    return base**(1.0 - val)


def shift_normalize(val, base=10**3):
    # TODO: implment check
    # it has to be defined with min=0.0
    shift_val = map_lin_to_log(0.0, base)
    val -= shift_val
    # it has to be defined with max=1.0
    norm_val = map_lin_to_log(1.0, base) - shift_val
    val /= norm_val
    return val


# def shape_function_lin(val): return 1-val


# TODO: try to figure out a good relationship between exp and magnitude_difference
def shape_function_exp(val, exp=CUST_MAGNITUDE): return (1-val)**exp


def evaluate_polynomial(x, coefs):
    val = 0.0
    for idx, coef in enumerate(coefs):
        val += coef * x**idx
    return val

def get_fitted_array (x, y, degree):
                
    # returns the fitted polynomial and the discrete array of displacements
    current_polynomial = np.poly1d(np.polyfit(x,y,degree))
    values = []
    for x_i in x:# evaluate the fitted eigenmode at certain intervals
        values.append(current_polynomial(x_i))
    eigenmodes_fitted = np.asarray(values)

    return eigenmodes_fitted 

def stop_run():
        stop = input('Want to continue? (y/n) ')
        if stop == 'n':
            sys.exit()