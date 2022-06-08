# --- External Imports ---
import scipy.optimize
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

    def GetEigenfrequencies(self,
                            boundary_conditions: tuple[str,str] = ("",""),
                            frequency_seeds: list[float] = [],
                            frequency_tolerance: float = 1e-2,
                            root_tolerance: float = 1e-10,
                            trivial_tolerance: float = 1e-3,
                            **kwargs) -> tuple[float]:
        """@brief Compute the (approximate) analytical eigenfrequencies of an Euler-Bernoulli beam.

           @details Newton iterations are performed around the passed seed points which are then filtered
                    with respect to a specified tolerance, then returned.
        """
        # Check input parameters
        if tuple(boundary_conditions) not in self.__boundary_conditions:
            raise ValueError("Invalid boundary conditions: {}. Options are: {}".format(boundary_conditions, self.__boundary_conditions))

        condition_matrix = []
        for condition_index, condition_type in enumerate(boundary_conditions):
            position = self.length * condition_index
            if condition_type == "fixed":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 0)(wave_number, position) for term_index in range(4)]) # displacement
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 1)(wave_number, position) for term_index in range(4)]) # angle
            elif condition_type == "free":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 2)(wave_number, position) for term_index in range(4)]) # moment
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 3)(wave_number, position) for term_index in range(4)]) # shear
            elif condition_type == "pinned":
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 0)(wave_number, position) for term_index in range(4)]) # displacement
                condition_matrix.append([lambda wave_number, position=position, term_index=term_index: self.__GetShapeTerm(term_index, 2)(wave_number, position) for term_index in range(4)]) # moment
            else:
                raise ValueError("Invalid boundary condition type: {}".format(condition_type))

        determinant = lambda argument: numpy.linalg.det([[term(argument) for term in row] for row in condition_matrix])

        eigenfrequencies = []
        for frequency_seed in frequency_seeds:
            wave_number = math.sqrt(frequency_seed * math.sqrt(self.section_density / self.stiffness / self.moment_of_inertia))

            # Do a couple of Newton iterations on the condition matrix' determinant to get the wave number
            try:
                wave_number = scipy.optimize.newton(determinant,
                                                    wave_number,
                                                    disp = False,
                                                    **kwargs)
            except:
                pass

            # Filter trivial solutions and failed newton iterations
            if trivial_tolerance < wave_number and abs(determinant(wave_number)) < root_tolerance:
                circular_frequency = math.sqrt(self.stiffness * self.moment_of_inertia / self.section_density) * wave_number**2
                if not any(abs((f - circular_frequency) / f) < frequency_tolerance for f in eigenfrequencies):
                    eigenfrequencies.append(circular_frequency)

        eigenfrequencies.sort()
        return eigenfrequencies
