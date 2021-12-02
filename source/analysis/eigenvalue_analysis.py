import numpy as np
from scipy import linalg
import json
from os.path import join as os_join

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
from source.auxiliary import global_definitions as GD
import source.postprocess.plotter_utilities as plotter_utilities
import source.postprocess.writer_utilitites as writer_utilities
import source.postprocess.visualize_skin_model_utilities as visualize_skin_model_utilities
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

        self.comp_m = np.copy(self.structure_model.comp_m)

    def solve(self, check_matrix=False):

        self.structure_model.eigenvalue_solve()
        # normalize results
        # http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect19.d/IAST.Lect19.Slides.pdf
        # normalize - unit generalized mass - slide 23

        [rows, columns] = self.structure_model.eigen_modes_raw.shape

        eig_modes_norm = np.zeros((rows, columns))

        gen_mass_raw = np.zeros(columns)
        gen_mass_norm = np.zeros(columns)

        print("Generalized mass should be identity")
        for i in range(len(self.structure_model.eig_values_raw)):
            gen_mass_raw[i] = np.matmul(np.matmul(np.transpose(self.structure_model.eigen_modes_raw[:, i]),
                                                  self.comp_m), self.structure_model.eigen_modes_raw[:, i])

            unit_gen_mass_norm_fact = np.sqrt(gen_mass_raw[i])

            eig_modes_norm[:, i] = self.structure_model.eigen_modes_raw[:,
                                                                        i]/unit_gen_mass_norm_fact

            gen_mass_norm[i] = np.matmul(np.matmul(np.transpose(eig_modes_norm[:, i]),
                                                   self.comp_m), eig_modes_norm[:, i])
            # print("norm ", i, ": ",gen_mass_norm[i])

        if check_matrix:
            gen_mass_norm = np.matmul(np.matmul(np.transpose(
                eig_modes_norm), self.comp_m), eig_modes_norm)
            print("Multiplication check: thethaT dot M dot theta: ",
                  gen_mass_norm, " numerically 0 for off-diagonal terms")
            print()

        self.eigenform = np.zeros(eig_modes_norm.shape)
        self.frequency = np.zeros(self.structure_model.eig_freqs.shape)
        self.period = np.zeros(self.structure_model.eig_pers.shape)

        for index in range(len(self.structure_model.eig_freqs)):
            self.eigenform[:, index] = eig_modes_norm[:,
                                                      self.structure_model.eig_freqs_sorted_indices[index]]
            self.frequency[index] = self.structure_model.eig_freqs[self.structure_model.eig_freqs_sorted_indices[index]]
            self.period[index] = self.structure_model.eig_pers[self.structure_model.eig_freqs_sorted_indices[index]]

        self.eigenform = self.structure_model.recuperate_bc_by_extension(
            self.eigenform)
        
        self.structure_model.eigenform_from_eigenvalue_analysis = self.eigenform

    def write_eigenmode_summary(self, global_folder_path, considered_modes=15):
        # TODO check to avoid redundancy in EigenvalueAnalysis and StructureModel
        # TODO remove code duplication: considered_modes
        if considered_modes == 'all':
            considered_modes = len(self.structure_model.dofs_to_keep)
        else:
            if considered_modes > len(self.structure_model.dofs_to_keep):
                considered_modes = len(self.structure_model.dofs_to_keep)

        file_header = '# Result of eigenvalue analysis\n'
        file_header += '# Mode | Eigenfrequency [Hz] | Period [s]\n'
        file_name = 'eigenvalue_analysis_eigenmode_analysis.dat'

        lines = []
        for idx in range(considered_modes):
            lines.append(
                [str(idx+1), '{:.5f}'.format(self.frequency[idx]), '{:.5f}'.format(self.period[idx])])

        writer_utilities.write_table(os_join(global_folder_path, file_name),
                                     file_header,
                                     lines)

    def plot_eigenmode_summary(self, pdf_report, display_plot, considered_modes=15):
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
        plot_title += 'Mode | Eigenfrequency [Hz] | Period [s]'

        row_labels = None
        column_labels = ['Mode', 'Eigenfrequency [Hz]', 'Period [s]']

        plotter_utilities.plot_table(pdf_report,
                                     display_plot,
                                     plot_title,
                                     table_data,
                                     row_labels,
                                     column_labels)

    def write_eigenmode_identification(self, global_folder_path):
        # TODO check to avoid redundancy in EigenvaluAnalysis and StructureModel

        lines = []
        counter = 0
        for mode_type, type_results in self.structure_model.mode_identification_results.items():
            type_counter = 0

            # type_results is an ordered list
            for t_res in type_results:
                m_id = t_res['mode_id']
                eff_mass = t_res['eff_modal_mass']
                rel_part = t_res['rel_participation']

                type_counter += 1
                counter += 1
                lines.append([str(counter),
                              str(m_id),
                              str(type_counter),
                              '{:.3f}'.format(
                                  self.structure_model.eig_freqs[self.structure_model.eig_freqs_sorted_indices[m_id-1]]),
                              mode_type,
                              '{:.3f}'.format(eff_mass),
                              '{:.3f}'.format(rel_part)])

        file_header = '# Result of decoupled eigenmode identification for the first ' + \
            str(counter) + ' mode(s)\n'
        file_header += '# ConsideredModesCounter | Mode | TypeCounter | Eigenfrequency [Hz] | Type |'
        file_header += ' EffModalMass [kg] or [kg*m^2] | RelPart | EffModalMass/TotalMass\n'

        file_name = 'eigenvalue_analysis_eigenmode_identification.dat'

        writer_utilities.write_table(os_join(global_folder_path, file_name),
                                     file_header,
                                     lines)

    def plot_eigenmode_identification(self, pdf_report, display_plot):
        # TODO check to avoid redundancy in EigenvaluAnalysis and StructureModel
        table_data = []
        counter = 0
        for mode_type, type_results in self.structure_model.mode_identification_results.items():
            type_counter = 0

            # type_results is an ordered list
            for t_res in type_results:
                m_id = t_res['mode_id']
                eff_mass = t_res['eff_modal_mass']
                rel_part = t_res['rel_participation']

                type_counter += 1
                counter += 1
                table_data.append([str(counter),
                                   str(m_id),
                                   str(type_counter),
                                   '{:.3f}'.format(
                                       self.structure_model.eig_freqs[self.structure_model.eig_freqs_sorted_indices[m_id-1]]),
                                   mode_type,
                                   '{:.3f}'.format(eff_mass),
                                   '{:.3f}'.format(rel_part)])

        plot_title = 'Result of decoupled eigenmode identification for the first ' + \
            str(counter) + ' mode(s)\n'

        row_labels = None
        column_labels = ['ConsideredModes',
                         'Mode',
                         'TypeCounter',
                         'Eigenfrequency\n [Hz]',
                         'Type',
                         'EffModalMass\n [kg] or [kg*m^2]',
                         'RelPart\n EffModalMass/TotalMass']

        plotter_utilities.plot_table(pdf_report,
                                     display_plot,
                                     plot_title,
                                     table_data,
                                     row_labels,
                                     column_labels)

    def get_output_for_visualiser(self):
        """"
        This function writes out the nodal dofs of the deformed state for visualiser
        """
        output = {}
        for key, val in self.structure_model.nodal_coordinates.items():
            output[key] = val.tolist()

        return output

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

        # nullify close to zero values
        # TODO: add to multiple places
        self.eigenform[abs(self.eigenform) < GD.THRESHOLD] = 0.0

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
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

    def write_selected_eigenmode(self, global_folder_path, selected_mode):
        """
        Pass to plot function:
            from structure model undeformed geometry
            self.eigenform -> as displacement  
            self.frequency -> in legend
            self.period -> in legend

        """
        selected_mode = selected_mode - 1

        print("Plotting result for a selected eigenmode in EigenvalueAnalysis \n")

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
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

        writer_utilities.write_result(os_join(global_folder_path, file_name), file_header,
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

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
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
            0, self.period[selected_mode], time_steps))  # AK: can this be called an array time ?

        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform.shape[0] + idx - step
            self.structure_model.nodal_coordinates[label] = self.eigenform[start:stop +
                                                                           1:step][:, selected_mode][:, np.newaxis] * array_time

        # for remaining dofs - case of 2D - create placeholders in correct format
        remaining_labels = list(set(GD.DOF_LABELS['3D'])-set(
            GD.DOF_LABELS[self.structure_model.domain_size]))
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

        plot_title = "Eigenmode: " + str(selected_mode+1)
        plot_title += "  Frequency: " + \
            '{0:.2f}'.format(self.frequency[selected_mode])
        plot_title += "  Period: " + \
            '{0:.2f}'.format(self.period[selected_mode])

        plotter_utilities.animate_result(plot_title,
                                         array_time,
                                         geometry,
                                         force,
                                         scaling)

    def animate_skin_model_for_selected_eigenmode(self, mode, skin_model_params):
        skin_model_params["result_path"] = os_join(
            "output", self.structure_model.name)
        skin_model_params["eigenvalue_analysis"] = {}
        skin_model_params["eigenvalue_analysis"]["mode"] = str(mode)
        skin_model_params["eigenvalue_analysis"]["frequency"] = self.frequency[mode]
        skin_model_params["eigenvalue_analysis"]["period"] = self.period[mode]
        skin_model_params["dofs_input"] = self.get_output_for_visualiser()

        visualize_skin_model_utilities.visualize_skin_model(skin_model_params)

    def postprocess(self, global_folder_path, pdf_report, display_plot, skin_model_params):
        """
        Postprocess something
        """
        print("Postprocessing in EigenvalueAnalysis derived class \n")

        if self.parameters['output']['eigenmode_summary']['write']:
            self.write_eigenmode_summary(global_folder_path)

        if self.parameters['output']['eigenmode_summary']['plot']:
            self.plot_eigenmode_summary(pdf_report, display_plot)

        if self.parameters['output']['eigenmode_identification']['write']:
            self.write_eigenmode_identification(global_folder_path)

        if self.parameters['output']['eigenmode_identification']['plot']:
            self.plot_eigenmode_identification(pdf_report, display_plot)

        for mode in self.parameters['output']['selected_eigenmode']['plot_mode']:
            self.plot_selected_eigenmode(pdf_report, display_plot, mode)

        for mode in self.parameters['output']['selected_eigenmode']['write_mode']:
            self.write_selected_eigenmode(global_folder_path, mode)

        if display_plot:
            for mode in self.parameters['output']['selected_eigenmode']['animate_mode']:
                self.animate_selected_eigenmode(mode)

        if skin_model_params is not None:
            for mode in self.parameters['output']['selected_eigenmode']['animate_skin_model']:
                self.animate_skin_model_for_selected_eigenmode(
                    mode, skin_model_params)

        # TODO to adapt and refactor
        # eigenvalue_analysis.plot_selected_first_n_eigenmodes(4)

        pass
