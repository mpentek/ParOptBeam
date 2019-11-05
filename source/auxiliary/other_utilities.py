from os.path import sep as os_sep


def get_adjusted_path_string(path_string):

    for separator in ['\\\\', '\\', '/', '//']:
        path_string = path_string.replace(separator, os_sep)

    return path_string[:]
