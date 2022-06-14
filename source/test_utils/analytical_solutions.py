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
