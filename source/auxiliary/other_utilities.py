from os.path import sep as os_sep


def get_adjusted_path_string(path_string):

    for separator in ['\\\\', '\\', '/', '//']:
        path_string = path_string.replace(separator, os_sep)

    return path_string[:]

import source.auxiliary.global_definitions as GD


def get_adjusted_path_string(path_string):

    for separator in ['\\\\', '\\', '/', '//']:
        path_string = path_string.replace(separator, os_sep)

    return path_string[:]

def get_signed_maximum(array):
    absolute_maximum = max(abs(array))
    max_id = list(abs(array)).index(absolute_maximum)
    
    return array[max_id]


def prepare_string_for_latex(string):
    greek = GD.GREEK
    if '_' in string:
        var, label = string.split('_')[0], string.split('_')[1]
        latex = r'${}$'.format(var) + r'$_{{{}}}$'.format(greek[label])
        #return string.replace('_','')
        return latex
    else:
        return string

