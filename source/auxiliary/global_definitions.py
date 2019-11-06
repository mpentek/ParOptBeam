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
    '3D': {'\"fixed-fixed\"': [[0, 1, 2, 3, 4, 5, -6], [0, 1, 2, 3, 4, 5]],
           '\"pinned-pinned\"': [[0, 1, 2], [3, 4, 5]],
           '\"fixed-pinned\"': [[0, 1, 2, 3, 4, 5], [3, 4, 5]],
           '\"pinned-fixed\"': [[0, 1, 2], [0, 1, 2, 3, 4, 5]],
           '\"fixed-free\"': [[0, 1, 2, 3, 4, 5]],
           '\"free-fixed\"': [[0, 1, 2, 3, 4, 5]]
           }}

BC_ELEMENT = {
     '\"fixed-fixed\"': [0, -1],
     '\"pinned-pinned\"': [0, -1],
     '\"fixed-pinned\"': [0, -1],
     '\"pinned-fixed\"': [0, -1],
     '\"fixed-free\"': [0],
     '\"free-fixed\"': [-1]
    }


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

# using these as default or fallback settings
DEFAULT_SETTINGS = {
    "name": "this_model_name",
    "domain_size": "3D",
    "system_parameters": {},
    "boundary_conditions": "fixed-free",
    "elastic_fixity_dofs": {}}
