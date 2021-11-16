These scripts can be used to evaluate the numerical calculation of (Cross) Power Spectral Densities. 
Some input parameters for the Python functions are evaluated and tested.

PSD detailed discussion and comparison
FftPsdEx -> plot_wind_spectrum_detailed_comparison_1 -> def get_velocity_spectra_methods

PSD final version (mostly the previous cleaned and consolidated) 
PsdComp -> plot_wind_spectrum_detailed_comparison_2-> def get_velocity_spectra_methods

With optimization:
Another method to fit a curve in a spectrum using optimization ->
CalcLuxTest -> calc_lux_methods.py -> def calculate_length_scale_criaciv_unipd -> my_func is optimized (assumes a von Karman spectrum, the unknown in this spectrum is the turbulence length scale, which it is solved for)
