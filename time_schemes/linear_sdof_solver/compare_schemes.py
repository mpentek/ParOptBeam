from itertools import cycle
import matplotlib.pyplot as plt
import sys
plt.rc('font',family='TeX Gyre Termes')
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)

USE_TWO_VARIABLE_FORMULATOIN = True
USE_ADAPTIVE_TIME_STEP = True

if USE_TWO_VARIABLE_FORMULATOIN == True:
    sys.path.append('two_variables')
else:
    sys.path.append('one_variable')

if USE_ADAPTIVE_TIME_STEP == True:
    from adaptive_sdof_solver import SDoF
else:
    from sdof_solver import SDoF

cycol = cycle('bgrcmk')
fig, axes = plt.subplots(4,1,figsize = (16,10))
#time_schemes = ["analytical", "euler", "bdf1", "bdf2", "rk4"]
time_schemes = ["analytical", "bdf2"]

def plot(sdof, time_scheme):
    global axes, cycol

    color = next(cycol)

    t, u, v = sdof.solve(time_scheme)
    eu, ev = sdof.error_estimation(t, u, v)

    axes[0].set_title(r"$Mu''(t) + Cu'(t) + Ku(t) = f(t), u(0) = 1, v(0) = 0$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u(t)")
    axes[0].grid('on')
    axes[0].plot(t, u, c = color)

    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r'$\varepsilon_u(t)$')
    axes[1].grid('on')
    axes[1].plot(t, eu, c = color)

    axes[2].set_xlabel("t")
    axes[2].set_ylabel("v(t)")
    axes[2].grid('on')
    axes[2].plot(t, v, c = color)

    axes[3].set_xlabel("t")
    axes[3].set_ylabel(r'$\varepsilon_v(t)$')
    axes[3].grid('on')
    axes[3].plot(t, ev, label=time_scheme, c = color)
    axes[3].legend()
    lgd = axes[3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.75), fancybox=True,ncol=len(time_schemes))


for time_scheme in time_schemes:
    sdof = SDoF(time_scheme)
    plot(sdof, time_scheme)
plt.show()
#plt.savefig("post_processing_results/two_varaibles.png")