AVAILABLE_BCS = ['\"fixed-fixed\"', '\"pinned-pinned\"', '\"fixed-pinned\"',
                 '\"pinned-fixed\"', '\"fixed-free\"', '\"free-fixed\"']

BC_DOFS = {
    '2D': {'\"fixed-fixed\"': [0, 1, 2, -3, -2, -1],
           '\"pinned-pinned\"': [0, 1, -5, -4],
           '\"fixed-pinned\"': [0, 1, 2, -5, -4],
           '\"pinned-fixed\"': [0, 1, -3, -2, -1],
           '\"fixed-free\"': [0, 1, 2],
           '\"free-fixed\"': [-3, -2, -1]
           },
    '3D': {'\"fixed-fixed\"': [0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1],
           '\"pinned-pinned\"': [0, 1, 2, -6, -5, -4],
           '\"fixed-pinned\"': [0, 1, 2, 3, 4, 5, -6, -5, -4],
           '\"pinned-fixed\"': [0, 1, 2, -6, -5, -4, -3, -2, -1],
           '\"fixed-free\"': [0, 1, 2, 3, 4, 5],
           '\"free-fixed\"': [-6, -5, -4, -3, -2, -1]
           }}

DOFS_PER_NODE = {'2D': 3,
                 '3D': 6}

DOF_LABELS = {'2D': ['x', 'y', 'g'],
              '3D': ['x', 'y', 'z', 'a', 'b', 'g']}

MODE_CATEGORIZATION = {
    '2D': {
        'longitudinal': ['x'],
        'sway_y': ['z', 'b']},
    '3D': {
        'longitudinal': ['x'],
        'torsional': ['a'],
        'sway_y': ['z', 'b'],
        'sway_z': ['y', 'g']}}

NODES_PER_LEVEL = 2

THRESHOLD = 1e-8


MODE_CATEGORIZATION_REVERSE = {'3D':{'x':'longitudinal',
                                      'a':'torsional',
                                      'z':'sway_y',
                                      'b':'sway_y',
                                      'y':'sway_z',
                                      'g':'sway_z'}}

CAARC_MODE_DIRECTIONS = {'0_deg':{'sway_z':0, 'sway_y':1, 'trosional':2}}

RESPONSE_DIRECTION_MAP = {'Qx':'x', 'Qy':'y', 'Qz':'z', 'Mx':'a', 'My':'b', 'Mz':'g'}
DIRECTION_RESPONSE_MAP = {'x':'Qx', 'y':'Qy', 'z':'Qz', 'a':'Mx', 'b':'My', 'g':'Mz'}

LOAD_DIRECTION_MAP = {'all':['y', 'z', 'a', 'b', 'g'],'Fy':'y', 'Fz':'z', 'Mx':'a', 'My':'b', 'Mz':'g'}# b = cross wind, g = anlong wind
DIRECTION_LOAD_MAP = {'x':'Fx', 'y':'Fy', 'z':'Fz', 'a':'Mx', 'b':'My', 'g':'Mz'}

UNITS_LOAD_DIRECTION = {'x':'[N/m]', 'y':'[N/m]', 'z':'[N/m]', 'a':'[Nm/m]', 'b':'[Nm/m]', 'g':'[Nm/m]'}

UNITS_POINT_LOAD_DIRECTION = {'x':'[N]', 'y':'[N]', 'z':'[N]', 'a':'[Nm]', 'b':'[Nm]', 'g':'[Nm]'}

UNIT_SCALE = {'N':1.0,'KN':1e-3,'MN':1e-6}

GREEK = {'y':'y','z':'z', 'x':'x','a':r'\alpha', 'b':r'\beta', 'g':r'\gamma'}

LOAD_DIRECTIONS_RESPONSES_UNCOUPLED = {'Qx':['x'], 'Qy':['y'], 'Qz':['z'], 'Mx':['a'], 'My':['z','b'], 'Mz':['y','g']}

LOAD_DIRECTIONS_RESPONSES_UNCOUPLED_RED = {'Qx':['x'], 'Qy':['y'], 'Qz':['z'], 'Mx':['a'], 'My':['z'], 'Mz':['y']}

LOAD_DIRECTIONS_RESPONSES_COUPLED = {'Qx':['x'], 'Qy':['y'], 'Qz':['z'], 'Mx':['y','a'], 'My':['z','b'], 'Mz':['y','g','a']}
