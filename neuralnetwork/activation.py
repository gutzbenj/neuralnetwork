from math import exp


def sigmoid(x) -> float:
    return 1 / (1 + exp(-x))
