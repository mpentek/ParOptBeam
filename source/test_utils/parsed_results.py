# --- External Imports ---
import numpy

# --- STL Imports ---
import pathlib


class ParsedResults:
    """A decorated dictionary containing the output of any implemented analysis."""
    
    def __init__(self, filePath: pathlib.Path):
        self._data = []
        self._file_path = pathlib.Path("")
        self._description = ""
        self._data_tags = []
        self._data_types = []

        self.Load(filePath)


    @property
    def data_types(self):
        return {
            "Type" : str
        }


    @property
    def description(self):
        return self._description


    @property
    def tags(self):
        return self._data_tags


    @property
    def data(self):
        return self._data


    def AsDictionary(self, include_description=True):
        dictionary = {}

        if include_description:
            dictionary["description"] = self._description

        for tag, items in zip(self._data_tags, self._data):
            dictionary[tag] = items

        return dictionary


    def Load(self, filePath: pathlib.Path):
        # Clear previous contents
        self._Clear()

        # Set output file path to be loaded
        argument_type = type(filePath)
        if not issubclass(argument_type, (pathlib.Path)):
            if issubclass(argument_type, (str)):
                filePath = pathlib.Path(filePath)
            else:
                raise ValueError("Expecting a pathlib.Path object for output file path, got {}".format(argument_type))
        self._file_path = filePath

        if self._file_path.is_file():
            with open(self._file_path, 'r') as file:

                # Parse description and data tags
                try:
                    self._ParseHeader(file)
                except Exception as exception:
                    raise RuntimeError("While parsing the header of {}:\n{}".format(self._file_path, exception))

                # Prase data
                try:
                    self._ParseData(file)
                except Exception as exception:
                    raise RuntimeError("While parsing data in {}:\n{}".format(self._file_path, exception))

        else: # self._file_path.is_file()
            raise FileNotFoundError("{} is not a file".format(self._file_path))


    def _Clear(self):
        self._data = []
        self._file_path = pathlib.Path("")
        self._description = ""
        self._data_tags = []
        self._data_types = []


    def _ParseHeader(self, file):
        """
        The header is expected in the following format:
        - all lines must begin with '#'
        - begins with any number of lines of description
        - ends with a line of data tags separated by '|'
        """
        header_end = file.tell()
        last_line = ""
        line = file.readline()

        while line:
            line = line.strip()

            if line and line[0] == '#': # header line
                if last_line != "":
                    self._description += last_line + '\n'
                last_line = line[1:].strip()
                header_end = file.tell()
            else: # non-header line
                if self._data_tags: # found and parsed data tags, header parsing can be finished
                    break
                else: # data tags have not been found
                    try:
                        self._ParseDataTags(last_line)
                    except Exception as exception:
                        raise SyntaxError("Expecting a line with data tags, got: {}\n{}".format(last_line,exception))

            line = file.readline()

        # Pop trailing newline from the description
        if self._description:
            self._description = self._description[:-1]

        # Reset the file pointer to data begin
        file.seek(header_end)


    def _ParseDataTags(self, dataTagLine: str):
        """
        Expecting one line of csv with '|' as delimiter. Example:
        tag_0 | tag_1 | tag_2|tag_3 |tag_4| tag_5 | ... | tag_n

        All data is interpreted as float, unless their tag has a type entry
        in ParsedResults::data_types
        """
        for tag in dataTagLine.split('|'):
            tag = tag.strip()

            if not (tag in self._data_tags):
                self._data_tags.append(tag)

                # Get data type based on its tag (default is float)
                if not tag in self.data_types:
                    self._data_types.append(float)
                else:
                    self._data_types.append(self.data_types[tag])

                self._data.append([])
            else:
                raise SyntaxError("Duplicate data tag: {}".format(tag))


    def _ParseData(self, file, delimiter=' '):
        """Data is expected in csv format"""
        # Extract data as strings
        for line in file:
            if line:
                items = line.strip().split(delimiter)

                if len(items) != len(self._data_tags):
                    raise SyntaxError("in line: {}\nnumber of items ({}) does not match the number of expected items ({})".format(line, len(items), len(self._data_tags)))

                for index, item in enumerate(items):
                    self._data[index].append(item.strip())

        # Convert numeric columns
        for index in range(len(self._data)):
            type = self._data_types[index]
            if type == float:
                self._data[index] = numpy.asarray([type(item) for item in self._data[index]])
            else:
                self._data[index] = [type(item) for item in self._data[index]]


    def __str__(self):
        # Header
        string = '# ' + self._description.replace('\n', "\n# ") + '\n# '

        for tag_index, tag in enumerate(self._data_tags):
            tmp = tag
            if tag_index < len(self._data_tags):
                tmp += " | "
            string += tmp
        string += '\n'

        # Data
        if self._data:
            for index in range(len(self._data[0])):
                tmp = ""
                for item in [self._data[component_index][index] for component_index in range(len(self._data))]:
                    tmp += str(item) + ' '
                string += tmp + '\n'

        return string


    def __getitem__(self, tag: str):
        """overload operator[] to function as a dict"""
        try:
            index = self._data_tags.index(tag)
            return self._data[index]
        except Exception as exception:
            raise KeyError("'{}' does not match any tags".format(tag))


    def keys(self):
        for key in self._data_tags:
            yield key


    def values(self):
        for value in self._data:
            yield value


    def items(self):
        for key, value in zip(self.keys(), self.values()):
            yield key, value