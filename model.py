
def normal_pr():
    return range(120, 200, 1)

def normal_st():
    return range(315, 325, 1)

def normal_qt():
    return range(300, 420, 1)

def normal_p():
    return [80]

def normal_t():
    return [160]


def calculate_mark(value, normal, i=100):
    value = int(value)
    if value > max(normal):
        return (max(normal) - value)
    if value < min(normal):
        return (min(normal) - value)
    else:
        return 0


def predict_for_ECG(pr, st, qt, p_tooth, t_tooth):
    result = {'pr': 0, 'st': 0, 'qt': 0, 'p_tooth': 0, 't_tooth': 0, 'result': 0}

    if pr in normal_pr():
        result['pr'] = 100
    else:
        result['pr'] = calculate_mark(pr, normal_pr(), i=10)
    return result