


def calc1(steps, stepsize, initial_value):
    tmp = []
    for x in range(steps):
        for y in range(initial_value, stepsize):
            for point in range(100):
                tmp.append(x*y)


    return tmp