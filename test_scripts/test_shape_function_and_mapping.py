import numpy as np
import matplotlib.pyplot as plt


def map_lin_to_log(val, base):
    # it has to be defined with a min=0.0 and max=1.0
    return base**(1.0 - val)


def shift_normalize(val, base):
    # it has to be defined with min=0.0
    shift_val = map_lin_to_log(0.0, base)
    val -= shift_val
    # it has to be defined with max=1.0
    norm_val = map_lin_to_log(1.0, base) - shift_val
    val /= norm_val
    return val


def shape_function_lin(val): return 1-val


def shape_function_exp(val, exp): return (1-val)**exp


# TODO: implement checks for min=0.0 and max=1.0
# 0.0 pure shear -> less values
# 1.0 pure bending -> more values
my_exp = 3
my_power = 2
my_base = 10**my_power

alpha = np.linspace(0.0, 1.0, 25)

my_mapped_val = [shift_normalize(
    map_lin_to_log(val, my_base), my_base) for val in alpha]

# linear shape functions for linearly spaced values
n1_s = [shape_function_lin(val) for val in alpha]
n1_b = [1-val for val in n1_s]

# exponential shape functions for log spaced values
n2_s = [shape_function_exp(val, my_exp) for val in my_mapped_val]
n2_b = [1-val for val in n2_s]


#
for my_val in np.linspace(0.0, 1.0, 10):
    print('my_val: ', my_val)
    print('n2_s: ', shape_function_exp(shift_normalize(
        map_lin_to_log(my_val, my_base), my_base), my_exp))
    print('n2_b: ', 1-shape_function_exp(shift_normalize(
        map_lin_to_log(my_val, my_base), my_base), my_exp))
    print('\n')
#


plt.figure(1)
plt.title('Linear spacing mapped to log')
plt.plot(alpha, alpha, marker='o', label='original')
plt.plot(alpha, my_mapped_val, marker='<', label='mapped')
plt.legend()

plt.figure(2)
plt.title('Linear spacing and linear shape functions')
plt.plot(alpha, n1_b, marker='o', label='n1_b')
plt.plot(alpha, n1_s, marker='<', label='n1_s')
plt.legend()

plt.figure(3)
plt.title('Lin-to-log spacing and exp shape functions')
plt.plot(my_mapped_val, n2_b, marker='o', label='n2_b')
plt.plot(my_mapped_val, n2_s, marker='<', label='n2_s')
plt.legend()

plt.figure(4)
plt.title('Lin-to-log spacing and exp shape functions')
plt.semilogy(my_mapped_val, n2_b, marker='o', label='n2_b')
plt.semilogy(my_mapped_val, n2_s, marker='<', label='n2_s')
plt.legend()

plt.show()
