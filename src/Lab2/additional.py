import enum


class PolynomialMethod(enum.Enum):
    SHIFTED_LEGANDRE = 1
    DOUBLED_SHIFTED_LEGANDRE = 2
    LEGANDRE = 3


class WeightMethod(enum.Enum):
    NORMED = 1
    MIN_MAX = 2


class LambdaMethod(enum.Enum):
    SINGLE_SET = 1
    TRIPLE_SET = 2
