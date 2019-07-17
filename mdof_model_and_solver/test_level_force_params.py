'''
These are the hardcoded test input values which were used for the level output
during the one-way-coupling simulation of the pylon

Here the test implies including endpoints implicitly that is why no extra checks are done
See the compute_level_force_process.py used as reference
'''
start_point_position = [0.0, 0.0, 0.0]
end_point_position = [0.0, 0.0, 68.03]
number_of_sampling_intervals = 25


# ===============================================================


# setup the parametric space for the internal points on the line
lower_bound = 0.0
upper_bound = 1.0

# initialize equidistant intervals
my_param = [1.0] * (number_of_sampling_intervals)
# overwrite first and last as endpoints are to be included
my_param[0] = 0.5
my_param[-1] = 0.5

parametrized_internal_points = [lower_bound + x*(upper_bound-lower_bound)/(
    number_of_sampling_intervals-1) for x in range(number_of_sampling_intervals + 1)]

parametrized_internal_points = [0.0] * (len(my_param)+1)

for i in range(len(my_param)):
    for j in range(i+1):
        parametrized_internal_points[i+1] += my_param[j]

parametrized_internal_points = [
    x / parametrized_internal_points[-1] for x in parametrized_internal_points]

print("Interval parameter: ", my_param)
print("Running parameter: ", parametrized_internal_points)
wait = input("check param...")

# determining the positions of the output points
direction_vector = [
    x - y for x, y in zip(end_point_position, start_point_position)]

current_positions = []
for k in range(len(parametrized_internal_points)):
    current_positions.append(
        [x + parametrized_internal_points[k]*y for x, y in zip(start_point_position, direction_vector)])

levels = {}
running_x_coords = []
for idx in range(len(current_positions)-1):
    levels[idx] = {}
    levels[idx]['start'] = current_positions[idx]
    levels[idx]['end'] = current_positions[idx+1]

    # first interval
    if idx == 0:
        levels[idx]['center'] = [
            x1 for x1 in levels[idx]['start']]
    elif idx == (len(current_positions)-1)-1:
        levels[idx]['center'] = [
            x2 for x2 in levels[idx]['end']]
    else:
        levels[idx]['center'] = [
            (x1+x2)/2 for x1, x2 in zip(levels[idx]['start'], levels[idx]['end'])]

    levels[idx]['output_file'] = None

    # the center is where the forces are computed
    # this has to correspond to the x-coords of the beam
    running_x_coords.append(levels[idx]['center'][2])

print('Levels: ', len(levels))
print(levels)


# ===============================================================


print('X coords: ', len(running_x_coords))
print(running_x_coords)
'''
Will result in the following 25 x-coords for the beam along the length

[0.0, 2.8345833333333332, 5.669166666666667, 8.50375, 11.338333333333335, 
14.172916666666666, 17.0075, 19.84208333333333, 22.67666666666667, 25.51125, 
28.34583333333333, 31.18041666666667, 34.015, 36.849583333333335, 39.68416666666667, 
42.51875, 45.35333333333334, 48.187916666666666, 51.0225, 53.857083333333335, 
56.69166666666666, 59.526250000000005, 62.36083333333333, 65.19541666666666, 68.03]

Has to correspond to the center values in the header of each level force file
Will represent the nodes in the 3D beam model
'''

ints = [y-x for x, y in zip(running_x_coords[:-1], running_x_coords[1:])]
print('Intervals: ', len(ints))
print(ints)
'''
Will result in the following 24 equidistant intervals for the beam along the length

[2.8345833333333332, 2.834583333333334, 2.834583333333333, 2.8345833333333346, 
2.834583333333331, 2.8345833333333346, 2.834583333333331, 2.834583333333338, 
2.834583333333331, 2.834583333333331, 2.834583333333338, 2.834583333333331, 
2.8345833333333346, 2.8345833333333346, 2.8345833333333275, 2.8345833333333417, 
2.8345833333333275, 2.8345833333333346, 2.8345833333333346, 2.8345833333333275, 
2.8345833333333417, 2.8345833333333275, 2.8345833333333275, 2.8345833333333417]
'''
wait = input('check values...')
