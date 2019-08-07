import numpy as np


def write_result(file_name, file_header, geometry, scaling):

    # default parameter
    # if reaction_force is None:
    #     reaction_force = np.zeros(1)
    # if force is None:
    #     force = np.zeros((len(displacement), n_data))
    #     force_scaling_factor=0.0

    # TEST GEOMETRY
    # print('TESTING GEOMETRY:')
    # print(geometry)
    # Set up figure
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')  # fig.add_subplot(111)

    # Handover data
    # x_undef = np.zeros(len(undeformed_geometry))
    # y_undef = undeformed_geometry
    # x_def = displacement*displacement_scaling_factor
    # x_force = force*force_scaling_factor

    # TODO: avoid try and except
    # use matrices straightforward
    try:
        # single / static case
        geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0],
                                geometry["deformation"][1] *
                                scaling["deformation"] +
                                geometry["undeformed"][1],
                                geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2]]
    except:
        # NOTE: now donw for 0-x, 1-y, 2-z DoFs
        # TODO: make consistent and generic
        geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis],
                                geometry["deformation"][1] * scaling["deformation"] +
                                geometry["undeformed"][1][:, np.newaxis],
                                geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis]]
        pass

    result_label = '# x0 x y z \n'
    lines = [['{:.8f}'.format(x0), '{:.8f}'.format(x), '{:.8f}'.format(y), '{:.8f}'.format(z)] for x0, x, y, z in zip(geometry["undeformed"][0],
                                                                            geometry["deformed"][0],
                                                                            geometry["deformed"][1],
                                                                            geometry["deformed"][2])]

    with open(file_name, "w") as output_file:
        output_file.write(file_header)
        output_file.write(result_label)
        for line in lines:
            output_file.write(' '.join(line) + '\n')

        output_file.close()

def write_result_at_dof(file_name, file_header, result, time):

    lines = [['{:.8f}'.format(t), '{:.8f}'.format(r)] for t, r in zip(time,result)]

    with open(file_name, "w") as output_file:
        output_file.write(file_header)
        for line in lines:
            output_file.write(' '.join(line) + '\n')

        output_file.close()
