import random
import scr.ml_utilities as mlu
from scr.ml_utilities import c
import scr.net_class
from scr.subject_tuple import Domain, SubjectTuple


def define_tuple():
    pass


def train_a():
    pass


def train_r():
    pass


def test_ar():
    pass


def understand():
    pass


if __name__ == '__main__':
    random.seed(c.RANDOM_SEED)
    domain = Domain(int, 7)
    subject_tuple = SubjectTuple(domain)
    print(subject_tuple.random())
