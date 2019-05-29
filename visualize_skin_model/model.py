import json

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Floor:
    def __init__(self, floor_geometry, height):
        """
        creating single floor based on the given floor geometry with the floor height
        """
        self.nodes = []
        self.x_vec, self.y_vec, self.z_vec = [],[],[] 
        for point in floor_geometry:
            x = point["x"]
            y = point["y"]
            z = height
            self.x_vec.append(x)
            self.y_vec.append(y)
            self.z_vec.append(z)
            node = Node(x, y, z)
            self.nodes.append(node)
        # adding the first point to close the geometry
        self.x_vec.append(self.x_vec[0])
        self.y_vec.append(self.y_vec[0])
        self.z_vec.append(self.z_vec[0])            


class Frame:
    def __init__(self, floor_geometry, height):
        """
        connecting the same points of a floor geometry for all floors
        """

class Structure:
    def __init__(self, structure_file):
        self.floors = []
        with open(structure_file) as json_file:
            data = json.load(json_file)
            self.floor_geometry = data["geometry"]
            self.structure_height = data["height"]
            self.num_of_floors = data["num_of_floors"]
            try:
                self.floor_height = data["floor_height"]
            except:
                self.floor_height = self.structure_height/self.num_of_floors
        
        self.print_structure_info()
        self.create_floors()

    def print_structure_info(self):    
        msg = "=============================================\n"
        msg += "MODEL INFO \n"
        msg += "HEIGHT:\t" + str(self.structure_height) + "\n"
        msg += "#FLOOR:\t" + str(self.num_of_floors) + "\n"
        msg += "FLOOR HEIGHT:\t" + str(self.floor_height)
        print(msg)

    def create_floors(self):
        current_height = 0.0
        while current_height < self.structure_height:
            current_height += self.floor_height  
            print(current_height)
            floor = Floor(self.floor_geometry, current_height)
            self.floors.append(floor)

if __name__ == "__main__":
    s = Structure("trapezoid.json")