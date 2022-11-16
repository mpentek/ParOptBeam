# --- External Imports ---
import scipy.optimize
import scipy.linalg
import numpy

# --- STL Imports ---
import math
import typing
import itertools


class EulerBernoulli:
    __boundary_conditions: list[str] = list(itertools.product(["fixed", "pinned", "free"], ["fixed", "pinned", "free"]))
    __shape_terms = (lambda wave_number, x: math.sinh(wave_number * x),
                     lambda wave_number, x: math.cosh(wave_number * x),
                     lambda wave_number, x: math.sin(wave_number * x),
                     lambda wave_number, x: math.cos(wave_number * x))

    def __init__(self,
                 section_density: float = 1.0,
                 stiffness: float = 1.0,
                 length: float = 1.0,
                 moment_of_inertia: float = 1.0):
        if section_density <= 0: raise ValueError("Invalid section_density: {}".format(section_density))
        if stiffness <= 0: raise ValueError("Invalid stiffness: {}".format(stiffness))
        if length <= 0: raise ValueError("Invalid length: {}".format(length))
        if moment_of_inertia <= 0: raise ValueError("Invalid moment of inertia: {}".format(moment_of_inertia))

        self.section_density = section_density
        self.stiffness = stiffness
        self.length = length
        self.moment_of_inertia = moment_of_inertia

    @staticmethod
    def __GetShapeTerm(term_index: int, derivative: int) -> typing.Callable:
        if term_index < 2:
            # Hyperbolic functions are each others' derivatives
            return lambda wave_number, x, term_index=term_index, derivative=derivative: EulerBernoulli.__shape_terms[(term_index + derivative) % 2](wave_number, x) * wave_number**derivative
        else:
            # Trigonometric functions have an extra sign cycle
            sign = -1 if (1 < ((term_index - 2 + derivative) % 4)) else 1
            return lambda wave_number, x, sign=sign, term_index=term_index, derivative=derivative: sign * EulerBernoulli.__shape_terms[2 + (term_index + derivative) % 2](wave_number, x) * wave_number**derivative

    def __GetConditionMatrix(self, boundary_conditions: tuple[str,str] = ("","")) -> list[list[typing.Callable]]:
        # Check input parameters
        if tuple(boundary_conditions) not in self.__boundary_conditions:
            raise ValueError("Invalid boundary conditions: {}. Options are: {}".format(boundary_conditions, self.__boundary_conditions))

        condition_matrix = []
        for condition_index, condition_type in enumerate(boundary_conditions):
            position = self.length * condition_index
            if condition_type == "fixed":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 0)(wave_number, position) for term_index in range(4)]) # displacement
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 1)(wave_number, position) for term_index in range(4)]) # rotation
            elif condition_type == "free":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 2)(wave_number, position) for term_index in range(4)]) # moment
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 3)(wave_number, position) for term_index in range(4)]) # shear
            elif condition_type == "pinned":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 0)(wave_number, position) for term_index in range(4)]) # displacement
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 2)(wave_number, position) for term_index in range(4)]) # moment
            else:
                raise ValueError("Invalid boundary condition type: {}".format(condition_type))

        return condition_matrix

    def __EigenfrequencyToWaveNumber(self, eigenfrequency: float) -> float:
        return math.sqrt(eigenfrequency * math.sqrt(self.section_density / self.stiffness / self.moment_of_inertia))

    def GetEigenfrequencies(self,
                            boundary_conditions: tuple[str,str] = ("",""),
                            frequency_seeds: list[float] = [],
                            frequency_tolerance: float = 1e-2,
                            root_tolerance: float = 1e-4,
                            trivial_tolerance: float = 1e0,
                            **kwargs) -> tuple[float]:
        """@brief Compute the (approximate) analytical eigenfrequencies of an Euler-Bernoulli beam.
           @details Newton iterations are performed around the passed seed points which are then filtered
                    with respect to a specified tolerance, then returned.
        """
        condition_matrix = self.__GetConditionMatrix(boundary_conditions)
        determinant = lambda argument: numpy.linalg.det([[term(argument) for term in row] for row in condition_matrix])

        eigenfrequencies = []
        for frequency_seed in frequency_seeds:
            wave_number = self.__EigenfrequencyToWaveNumber(frequency_seed)

            # Do a couple of Newton iterations on the condition matrix' determinant to get the wave number
            try:
                wave_number = scipy.optimize.newton(determinant,
                                                    wave_number,
                                                    disp = False,
                                                    **kwargs)
            except:
                pass

            # Filter trivial solutions and failed newton iterations
            angular_frequency = math.sqrt(self.stiffness * self.moment_of_inertia / self.section_density) * wave_number**2
            if trivial_tolerance < abs(angular_frequency) and abs(determinant(wave_number)) < root_tolerance:
                if not any(abs((f - angular_frequency) / f) < frequency_tolerance for f in eigenfrequencies):
                    eigenfrequencies.append(angular_frequency)

        eigenfrequencies.sort()
        return eigenfrequencies

    @property
    def position_samples(self) -> numpy.array:
        return numpy.linspace(0.0, self.length, 100)

    @property
    def mass(self) -> float:
        return self.section_density * self.length

    def __GetShapeTermCoefficients(self, wave_number: float, boundaries: tuple) -> list["float"]:
        wl = wave_number * self.length
        tmp_difference = (numpy.cosh(wl) - numpy.cos(wl)) / (numpy.sin(wl) - numpy.sinh(wl))
        tmp_sum = (numpy.cosh(wl) + numpy.cos(wl)) / (numpy.sin(wl) + numpy.sinh(wl))
        if boundaries.count("pinned") == 2:
            return [0.0, 0.0, 1.0, 0.0]
        elif boundaries.count("fixed") == 2:
            return [-tmp_difference, -1.0, tmp_difference, 1.0]
        elif boundaries.count("free") == 2:
            return [tmp_difference, 1.0, tmp_difference, 1.0]
        elif "fixed" in boundaries and "pinned" in boundaries:
            return [-tmp_difference, -1.0, tmp_difference, 1.0]
        elif "fixed" in boundaries and "free" in boundaries:
            return [tmp_sum, -1.0, -tmp_sum, 1.0]
        else:
            raise ValueError(f"Unsupported boundary conditions: {boundaries}")

    def GetMode(self, eigenfrequency: float, boundary_conditions: tuple) -> typing.Callable:
        wave_number = self.__EigenfrequencyToWaveNumber(eigenfrequency)
        shape_term_coefficients = self.__GetShapeTermCoefficients(wave_number, boundary_conditions)
        def mode(position: float) -> float:
            value = 0.0
            for i_term, term_coefficient in enumerate(shape_term_coefficients):
                value += term_coefficient * self.__GetShapeTerm(i_term, 0)(wave_number, position)
            return value
        return mode

    def GetModalProperties(self, eigenfrequency: float, boundary_conditions: tuple) -> float:
        mode = self.GetMode(eigenfrequency, boundary_conditions)
        position_samples = self.position_samples
        mode_samples = numpy.array([mode(x) for x in position_samples])
        
        # # --- Debug begin ---
        # import numpy as np
        # import matplotlib.pyplot as plt
        # plt.plot(position_samples, mode_samples)
        # plt.show()
        # wait=input("check1")
        # plt.close()
        # # --- Debug end ---
        
        # # --- Debug begin ---
        # total_mass = 0.0
        # eff_modal_numerator_sum = 0.0
        # eff_modal_denominator_sum = 0.0
        
        # for i in range(len(position_samples)-1):
        #     storey_mass = (position_samples[i+1]-position_samples[i])*self.section_density
        #     total_mass += storey_mass
        #     # numerator like this 100.0
        #     #eff_modal_numerator_sum += (storey_mass * phi_n[floor][mode]) **2

        #     eff_modal_numerator_sum += (storey_mass * mode_samples[i])
        #     eff_modal_denominator_sum += storey_mass * mode_samples[i] ** 2

        #     # denominator always 1.0

        # # numerator like this NOT 100.0
        # eff_modal_numerator_sum = eff_modal_numerator_sum **2

        # eff_modal_mass_sum = eff_modal_numerator_sum / eff_modal_denominator_sum
        # rel_participation_sum = eff_modal_mass_sum / total_mass
        # print(eff_modal_numerator_sum)
        # print(eff_modal_denominator_sum)
        # print(total_mass)
        # print(rel_participation_sum)
        # # --- Debug end ---
        
        eff_modal_numerator_sum = (numpy.trapz(mode_samples, position_samples) * self.section_density)**2
        eff_modal_denominator_sum = numpy.trapz(mode_samples**2, position_samples) * self.section_density
        eff_modal_mass_sum = eff_modal_numerator_sum / eff_modal_denominator_sum
        total_mass = (position_samples[-1]-position_samples[0])* self.section_density
        rel_participation_sum = eff_modal_mass_sum / total_mass
        # print(eff_modal_numerator_sum)
        # print(eff_modal_denominator_sum)
        # print(total_mass)
        # print(rel_participation_sum)
        
        return eff_modal_mass_sum, rel_participation_sum

    def GetDynamicSolution(self, initial_displacement: float, boundary_conditions: tuple["str","str"]) -> typing.Callable:
        if boundary_conditions[0] != "fixed" or boundary_conditions[1] != "free":
            raise RuntimeError("Unsupported boundary conditions")

        frequency_coefficient = numpy.sqrt(self.stiffness / self.section_density * self.moment_of_inertia / self.length**4)
        eigenfrequencies = [frequency_coefficient * root**2 for i, root in enumerate((1.875, 4.694, 7.885))]
        wave_numbers = [self.__EigenfrequencyToWaveNumber(eigenfrequency) for eigenfrequency in eigenfrequencies]

        def unscaled_initial_shape(position: float) -> float:
            return position**3 / 6.0 / self.stiffness / self.moment_of_inertia * (3.0 * self.length - position)

        initial_shape_coefficient = initial_displacement / unscaled_initial_shape(self.length)
        initial_shape = lambda x: initial_shape_coefficient * unscaled_initial_shape(x)

        def mode(wave_number: float, position: float) -> float:
            wl = wave_number * self.length
            wx = wave_number * position
            tmp = (numpy.cos(wl) + numpy.cosh(wl)) / (numpy.sin(wl) + numpy.sinh(wl))
            return numpy.cos(wx) - numpy.cosh(wx) - tmp * numpy.sin(wx) + tmp * numpy.sinh(wx)

        mode_amplitudes = []
        position_samples = numpy.linspace(0.0, self.length, 100)
        for wave_number in wave_numbers:
            integrand = [mode(wave_number, x) * initial_shape(x) for x in position_samples]
            integral = numpy.trapz(integrand, position_samples)
            mode_amplitudes.append(integral)

        def solution(time: float, position: float) -> float:
            value = 0.0
            for eigenfrequency, wave_number, mode_amplitude in zip(eigenfrequencies, wave_numbers, mode_amplitudes):
                value += mode_amplitude * mode(wave_number, position) * numpy.cos(eigenfrequency * time)
            return value

        return lambda t, x: solution(t, x) * initial_displacement / solution(0, self.length)


class TorsionalBeam:

    def __init__(self,
                 stiffness: float = 0.0,
                 poisson_ratio: float = 0.0,
                 density: float = 0.0,
                 length: float = 0.0,
                 moment_of_inertia_y: float = 0.0,
                 moment_of_inertia_z: float = 0.0,
                 torsional_moment_of_inertia: float = 0.0):
        self.stiffness = stiffness
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.length = length
        self.moment_of_inertia_y = moment_of_inertia_y
        self.moment_of_inertia_z = moment_of_inertia_z
        self.torsional_moment_of_inertia = torsional_moment_of_inertia

    def GetEigenfrequencies(self,
                            boundary_conditions: tuple[str,str] = ("",""),
                            number_of_modes: int = 0) -> list[float]:
        angular_frequencies = []
        mode_index = 0

        material_coefficient = math.sqrt(self.stiffness / 2.0 / (1.0 + self.poisson_ratio) / self.density)
        geometry_coefficient = math.sqrt(self.torsional_moment_of_inertia / (self.moment_of_inertia_y + self.moment_of_inertia_z)) / self.length

        # Free and pinned boundaries are interchangable in this case
        # Both sides fixed == both sides free
        if boundary_conditions[0] == boundary_conditions[1] or all(condition in ("free", "pinned") for condition in boundary_conditions):
            trigonometric_offset = 1.0
        else:
            trigonometric_offset = 0.5

        for mode_index in range(number_of_modes):
            angular_frequency = (mode_index + trigonometric_offset) * math.pi * material_coefficient * geometry_coefficient
            angular_frequencies.append(angular_frequency)

        return angular_frequencies
