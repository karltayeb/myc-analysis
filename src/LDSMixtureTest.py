import LDSMixture as LDSMixture
import numpy as np


def test1():
    model = LDSMixture.LDSMixture()
    data = (np.ones((3,3)) * np.arange(3)).T
    model.initialize(data, 3, 1, 1, 3, 1, 1e-5)
    model.estep(data)


def test2():
    model = LDSMixture.LDSMixture()
    
    data = np.random.random((50, 10))
    model.initialize(data, 3, 1, 1, 10, 1, 1e-8)
    model.em(data)

def test3():
    model = LDSMixture.LDSMixture()
    data = np.random.random((50, 10))
    model.initialize(data, 3, 1, 1, 10, 1e-8, 1)
    model.em(data)

def test4():
    model = LDSMixture.LDSMixture()
    data = np.random.random((50, 10))
    model.initialize(data, 3, 1, 1, 10, 1, 1)
    model.em(data)

if __name__ == "__main__":
    test3()