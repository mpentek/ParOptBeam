AVAILABLE_BCS = ['\"fixed-fixed\"', '\"pinned-pinned\"', '\"fixed-pinned\"',
                 '\"pinned-fixed\"', '\"fixed-free\"', '\"free-fixed\"']

BC_DOFS = {
    '2D': {'\"fixed-fixed\"': [0, 1, 2, -3, -2, -1],
           '\"pinned-pinned\"': [0, 1, -3, -2],
           '\"fixed-pinned\"': [0, 1, 2, -3, -2],
           '\"pinned-fixed\"': [0, 1, -3, -2, -1],
           '\"fixed-free\"': [0, 1, 2],
           '\"free-fixed\"': [-3, -2, -1]
           },
    '3D': {'\"fixed-fixed\"': [0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1],
           # torsion is considered fixed for anything involving "pinned"
           '\"pinned-pinned\"': [0, 1, 2, 3, -6, -5, -4, -3],
           '\"fixed-pinned\"': [0, 1, 2, 3, 4, 5, -6, -5, -4, -3],
           '\"pinned-fixed\"': [0, 1, 2, 3, -6, -5, -4, -3, -2, -1],
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
        'sway_z': ['y', 'g']},
    '3D': {
        'longitudinal': ['x'],
        'torsional': ['a'],
        'sway_y': ['z', 'b'],
        'sway_z': ['y', 'g']}}

NODES_PER_LEVEL = 2

THRESHOLD = 1e-8