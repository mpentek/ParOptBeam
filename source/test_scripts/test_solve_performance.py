
if __name__ == "__main__":
    '''
    A helper script to decide if solve performance is affected by
    np.linalg solve for matric-vector vs 
    fill-diagonal matrix -> vector-vector element-wise division
    with or without dimension (type) check

    '''

    import timeit

    size = 60*6
    it = 30000

    # version 1
    setup = '''
    import numpy as np

    n = %d

    my_matrix = np.random.rand(n,n)
    print(my_matrix.ndim)

    my_vector = np.random.rand(n)
    print(my_vector.ndim)
    ''' % (size)

    code = '''
    solve = np.linalg.solve(my_matrix, my_vector)
    '''

    t1 = timeit.timeit(stmt=code, setup=setup, number=it)
    print("Time: linalg solve - direct call " + str(t1))

    # version 2
    setup = '''
    import numpy as np

    n = %d

    my_vector = np.random.rand(n)
    print(my_vector.ndim)

    my_matrix = np.diag(np.random.rand(n,n))
    print(my_matrix.ndim)
    ''' % (size)

    code = '''
    solve = my_matrix/my_vector
    '''

    t2 = timeit.timeit(stmt=code, setup=setup, number=it)
    print("Time: div solve - direct call " + str(t2))

    # version 1 mod
    setup = '''
    import numpy as np

    n = %d

    my_matrix = np.random.rand(n,n)
    print(my_matrix.ndim)

    my_vector = np.random.rand(n)
    print(my_vector.ndim)
    ''' % (size)

    code = '''
    if my_matrix.ndim == 2:
        solve = np.linalg.solve(my_matrix, my_vector)
    else:
        solve = my_matrix/my_vector
    '''

    t1m = timeit.timeit(stmt=code, setup=setup, number=it)
    print("Time: linalg solve - if call " + str(t1m))

    # version 2
    setup = '''
    import numpy as np

    n = %d

    my_vector = np.random.rand(n)
    print(my_vector.ndim)

    my_matrix = np.diag(np.random.rand(n,n))
    print(my_matrix.ndim)
    ''' % (size)

    code = '''
    if my_matrix.ndim == 2:
        solve = np.linalg.solve(my_matrix, my_vector)
    else:
        solve = my_matrix/my_vector
    '''

    t2m = timeit.timeit(stmt=code, setup=setup, number=it)
    print("Time: div solve - if call " + str(t2m))

    """
    Console output:

    Time: linalg solve - direct call 15.431618875999902

    Time: div solve - direct call 0.0356253459999607


    Time: linalg solve - if call 15.642636252999637

    Time: div solve - if call 0.037879820999933145

    ==>> IF does not seem to make a noticable difference
    """