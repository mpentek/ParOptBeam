import numpy as np
from scipy import linalg
import json
from os.path import join, isdir
from os import makedirs

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.writer_utilitites as writer_utilities
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults


class EigenvalueAnalysis(AnalysisType):
    """
    Derived class for the (dynamic) eigenvalue analysis of a given structure model        
    """

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "type": "eigenvalue_analysis",
        "settings": {},
        "input": {},
        "output": {}}

    def __init__(self, structure_model, parameters, name="EigenvalueAnalysis"):
        # TODO: add number of considered modes for output parameters upper level
        # also check redundancy with structure model

        # validating and assign model parameters
        validate_and_assign_defaults(
            EigenvalueAnalysis.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        super().__init__(structure_model, self.parameters["type"])

        # adding additional attributes to the derived class
        self.eigenform = None
        self.frequency = None
        self.period = None

    def solve(self):

        k = self.structure_model.apply_bc_by_reduction(self.structure_model.k)
        m = self.structure_model.apply_bc_by_reduction(self.structure_model.m)

        eig_values_raw, eig_modes_raw = linalg.eigh(k, m)
        # rad/s
        eig_values = np.sqrt(np.real(eig_values_raw))
        # 1/s = Hz
        eig_freqs = eig_values / 2. / np.pi
        # s
        eig_pers = 1. / eig_freqs
        # sort eigenfrequencies
        eig_freqs_sorted_indices = np.argsort(eig_freqs)
        ##

        # normalize results
        # http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect19.d/IAST.Lect19.Slides.pdf
        # normalize - unit generalized mass - slide 23

        [rows, columns] = eig_modes_raw.shape

        eig_modes_norm = np.zeros((rows, columns))

        gen_mass_raw = np.zeros(columns)
        gen_mass_norm = np.zeros(columns)

        print("Generalized mass should be identity")
        for i in range(len(eig_values_raw)):
            gen_mass_raw[i] = (np.transpose(eig_modes_raw[:, i])).dot(
                m).dot(eig_modes_raw[:, i])

            unit_gen_mass_norm_fact = np.sqrt(gen_mass_raw[i])

            eig_modes_norm[:, i] = eig_modes_raw[:, i]/unit_gen_mass_norm_fact

            gen_mass_norm[i] = (np.transpose(eig_modes_norm[:, i])).dot(
                m).dot(eig_modes_norm[:, i])
            # print("norm ", i, ": ",gen_mass_norm[i])

        # print("Multiplication check: thethaT dot M dot theta: ",(np.transpose(eig_modes_norm)).dot(self.structure_model.m).dot(eig_modes_norm)," numerically 0 for off-diagonal terms")
        # print()

        self.eigenform = np.zeros(eig_modes_norm.shape)
        self.frequency = np.zeros(eig_freqs.shape)
        self.period = np.zeros(eig_pers.shape)

        for index in range(len(eig_freqs)):
            self.eigenform[:, index] = eig_modes_norm[:,
                                                      eig_freqs_sorted_indices[index]]
            self.frequency[index] = eig_freqs[eig_freqs_sorted_indices[index]]
            self.period[index] = eig_pers[eig_freqs_sorted_indices[index]]

        self.eigenform = self.structure_model.recuperate_bc_by_extension(
            self.eigenform)

    def write_eigenmode_summary(self, considered_modes=10):
        # TODO check to avoid redundancy in EigenvalueAnalysis and StructureModel
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.structure_model.dofs_to_keep)
        else:
            if considered_modes > len(self.structure_model.dofs_to_keep):
                considered_modes = len(self.structure_model.dofs_to_keep)

        file_header = '# Result of eigenvalue analysis\n'
        file_header += '# Mode Eigenfrequency [Hz] Period [s]\n'
        file_name = 'eigenvalue_analysis_eigenmode_analysis.dat'

        lines = []
        for idx in range(considered_modes):
            lines.append(
                [str(idx+1), '{:.5f}'.format(self.frequency[idx]), '{:.5f}'.format(self.period[idx])])

        absolute_folder_path = join(
            "output", self.structure_model.name)
        # make sure that the absolute path to the desired output folder exists
        if not isdir(absolute_folder_path):
            makedirs(absolute_folder_path)

        writer_utilities.write_table(join(absolute_folder_path, file_name),
                                     file_header,
                                     lines)

    def plot_eigenmode_summary(self, pdf_report, display_plot, considered_modes=10):
        # TODO check to avoid redundancy in EigenvalueAnalysis and StructureModel
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.structure_model.dofs_to_keep)
        else:
            if considered_modes > len(self.structure_model.dofs_to_keep):
                considered_modes = len(self.structure_model.dofs_to_keep)

        table_data = []
        for idx in range(considered_modes):
            table_data.append(
                [str(idx+1), '{:.5f}'.format(self.frequency[idx]), '{:.5f}'.format(self.period[idx])])

        plot_title = 'Result of eigenvalue analysis\n'
        plot_title += 'Mode Eigenfrequency [Hz] Period [s]'

        row_labels = None
        column_labels = ['Mode', 'Eigenfrequency [Hz]', 'Period [s]']

        plotter_utilities.plot_table(pdf_report,
                                     display_plot,
                                     plot_title,
                                     table_data,
                                     row_labels,
                                     column_labels)

    def write_eigenmode_identification(self):
        # TODO check to avoid redundancy in EigenvaluAnalysis and StructureModel

        lines = []
        counter = 0
        for mode, mode_ids in self.structure_model.mode_identification_results.items():
            type_counter = 0
            for mode_id in mode_ids:
                type_counter += 1
                counter += 1
                lines.append([str(counter), str(mode_id), str(type_counter), '{:.5f}'.format(
                    self.structure_model.eig_freqs[self.structure_model.eig_freqs_sorted_indices[mode_id-1]]), mode])

        file_header = '# Result of decoupled eigenmode identification for the first ' + \
            str(counter) + ' mode(s)\n'
        file_header += '# ConsideredModesCounter Mode TypeCounter Eigenfrequency [Hz] Type\n'

        file_name = 'eigenvalue_analysis_eigenmode_identification.dat'
        absolute_folder_path = join(
            "output", self.structure_model.name)
        # make sure that the absolute path to the desired output folder exists
        if not isdir(absolute_folder_path):
            makedirs(absolute_folder_path)

        writer_utilities.write_table(join(absolute_folder_path, file_name),
                                     file_header,
                                     lines)

    def plot_eigenmode_identification(self, pdf_report, display_plot):
        # TODO check to avoid redundancy in EigenvaluAnalysis and StructureModel
        table_data = []
        counter = 0
        for mode, mode_ids in self.structure_model.mode_identification_results.items():
            type_counter = 0
            for mode_id in mode_ids:
                type_counter += 1
                counter += 1
                table_data.append([str(counter), str(mode_id), str(type_counter), '{:.5f}'.format(
                    self.structure_model.eig_freqs[self.structure_model.eig_freqs_sorted_indices[mode_id-1]]), mode])

        plot_title = 'Result of decoupled eigenmode identification for the first ' + \
            str(counter) + ' mode(s)\n'

        row_labels = None
        column_labels = ['ConsideredModesCounter', 'Mode',
                         'TypeCounter', 'Eigenfrequency [Hz]', 'Type']

        plotter_utilities.plot_table(pdf_report,
                                     display_plot,
                                     plot_title,
                                     table_data,
                                     row_labels,
                                     column_labels)

    # TODO: to remove once visualizer is refactored and tested
    def write_output_file(self):
        """"
        This function writes out the nodal dofs of the deformed state

        """
        file = open("beam.txt", "w")
        dict = {}
        dict["length"] = max(self.structure_model.nodal_coordinates["x0"])
        dict["num_of_elements"] = len(
            self.structure_model.nodal_coordinates["x0"])
        for key, val in self.structure_model.nodal_coordinates.items():
            dict[key] = val.tolist()

        json_string = json.dumps(dict)

        file.write(json_string)
        file.close()

    def plot_selected_eigenmode(self, pdf_report, display_plot, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend

        """
        selected_mode = selected_mode - 1

        print("Plotting result for a selected eigenmode in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode]

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = " Eigenmode: " + str(selected_mode+1)
        plot_title += "  Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode])
        plot_title += "  Period: " + \
            '{0:.2f}'.format(self.period[selected_mode])

        plotter_utilities.plot_result(pdf_report,
                                      display_plot,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      1)

    def write_selected_eigenmode(self, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend

        """
        selected_mode = selected_mode - 1

        print("Plotting result for a selected eigenmode in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode]

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}

        file_header = "# Eigenmode: " + str(selected_mode+1) + "\n"
        file_header += "# Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode]) + "\n"
        file_header += "# Period: " + \
            '{0:.2f}'.format(self.period[selected_mode]) + "\n"

        file_name = 'eigenvalue_analysis_selected_eigenmode_' + \
            str(selected_mode) + '.dat'
        absolute_folder_path = join(
            "output", self.structure_model.name)
        # make sure that the absolute path to the desired output folder exists
        if not isdir(absolute_folder_path):
            makedirs(absolute_folder_path)

        writer_utilities.write_result(join(absolute_folder_path, file_name), file_header,
                                      geometry, scaling)

    def plot_selected_first_n_eigenmodes(self, pdf_report, display_plot, number_of_modes):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend
        """

        print("Plotting result for selected first n eigenmodes in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop+1:step]

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = " "
        for selected_mode in range(number_of_modes):
            plot_title += "Eigenmode " + str(selected_mode + 1) + "  Frequency: " + str(np.round(
                self.frequency[selected_mode], 3)) + "  Period: " + str(np.round(self.period[selected_mode], 3)) + "\n"
        plotter_utilities.plot_result(pdf_report,
                                      display_plot,
                                      plot_title,
                                      geometry,
                                      force,
                                      scaling,
                                      number_of_modes)

    def animate_selected_eigenmode(self, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend  
        """
        selected_mode = selected_mode - 1

        print("Animating eigenmode in EigenvalueAnalysis \n")

        time_steps = 100
        array_time = np.sin(2 * np.pi * self.frequency[selected_mode] * np.linspace(
            0, self.period[selected_mode], time_steps))  # AK: can this be called an array tiem ?

        for idx, label in zip(list(range(StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size])),
                              StraightBeam.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = StraightBeam.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode][:, np.newaxis] * array_time

        # for remaining dofs - case of 2D - create placeholders in correct format
        remaining_labels = list(set(StraightBeam.DOF_LABELS['3D'])-set(
            StraightBeam.DOF_LABELS[self.structure_model.domain_size]))
        for label in remaining_labels:
            self.structure_model.nodal_coordinates[label] = np.zeros(
                (self.structure_model.n_nodes, len(array_time)))

        # NOTE: this should be the correct way
        # TODO: add some generic way to be able to subtract some non-zero origin point
        # TODO: check if an origin point shift or extension still needed

        geometry = {"undeformed": [self.structure_model.nodal_coordinates["x0"],
                                   self.structure_model.nodal_coordinates["y0"],
                                   self.structure_model.nodal_coordinates["z0"]],
                    "deformation": [self.structure_model.nodal_coordinates["x"],
                                    self.structure_model.nodal_coordinates["y"],
                                    self.structure_model.nodal_coordinates["z"]],
                    "deformed": None}

        force = {"external": None,
                 "base_reaction": None}

        scaling = {"deformation": 1,
                   "force": 1}

        plot_title = "Eigenmode: " + str(selected_mode)
        plot_title += "  Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode])
        plot_title += "  Period: " + \
            '{0:.2f}'.format(self.period[selected_mode])

        plotter_utilities.animate_result(plot_title,
                                         array_time,
                                         geometry,
                                         force,
                                         scaling)

    def postprocess(self, pdf_report, display_plot):
        """
        Postprocess something
        """
        print("Postprocessing in EigenvalueAnalysis derived class \n")

        if self.parameters['output']['eigenmode_summary']['write']:
            self.write_eigenmode_summary()

        if self.parameters['output']['eigenmode_summary']['plot']:
            self.plot_eigenmode_summary(pdf_report, display_plot)

        if self.parameters['output']['eigenmode_identification']['write']:
            self.write_eigenmode_identification()

        if self.parameters['output']['eigenmode_identification']['plot']:
            self.plot_eigenmode_identification(pdf_report, display_plot)

        for mode in self.parameters['output']['selected_eigenmode']['plot_mode']:
            self.plot_selected_eigenmode(pdf_report, display_plot, mode)

        for mode in self.parameters['output']['selected_eigenmode']['write_mode']:
            self.write_selected_eigenmode(mode)

        for mode in self.parameters['output']['selected_eigenmode']['animate_mode']:
            self.animate_selected_eigenmode(mode)

        # TODO to adapt and refactor
        # eigenvalue_analysis.plot_selected_first_n_eigenmodes(4)

        pass
