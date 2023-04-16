import numpy as np
from abc import ABC, abstractmethod
from .projection import ProjMat

# TODO Switch to sparseness
# TODO Two Sex matrix?


class KinshipModel(object):
    def __init__(self, proj_mat: ProjMat):
        self.U = proj_mat.UU
        self.ff = proj_mat.ff
        self.pp = proj_mat.mother_dist
        self.n = proj_mat.UU.shape[0]

        for key, KinClass in kins.items():
            kin_instance = KinClass(self)
            kin_instance.do_ageing()
            setattr(self, key, kin_instance)


class Kin(ABC):

    def __init__(self, kin_model: KinshipModel, k0):
        """
        :param kin_model: Instance of KinshipModel
        :param k0: Initial condition for this type of kin
        """
        self.k0 = k0
        self.kin_model = kin_model
        # kin_matrix has rows corresponding to
        self.kin_matrix = np.zeros((kin_model.n,
                                    kin_model.n))
        self.kin_matrix[0, :] = k0

    def do_survival(self, ka):
        kb = self.kin_model.U.dot(ka)
        return kb

    def do_ageing(self):
        for x in range(1, self.kin_model.n):
            self.kin_matrix[x, :] = self.do_survival(self.kin_matrix[x - 1, :])

    def get_kin_distribution(self, focal_age, kin_age=None):
        if kin_age is None:
            return self.kin_matrix[focal_age, :]
        elif focal_age is None:
            return self.kin_matrix[:, kin_age]
        else:
            return self.kin_matrix[focal_age, kin_age]


class SubsidisedKin(Kin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_ageing(self):
        for x in range(1, self.kin_model.n):
            self.kin_matrix[x, :] = self.do_survival(self.kin_matrix[x - 1, :])
            self.kin_matrix[x, 0] += self.do_subsidy(self.get_donor(x))

    @abstractmethod
    def get_donor(self, age):
        pass

    def do_subsidy(self, donor: np.array):
        return self.kin_model.ff.dot(donor)


class Daughter(Kin):
    def __init__(self, kin_model: KinshipModel):
        k0 = np.zeros(kin_model.n)
        super().__init__(kin_model, k0)

    def do_ageing(self):
        for x in range(1, self.kin_model.n):
            self.kin_matrix[x, :] = self.do_survival(self.kin_matrix[x - 1, :])
            self.kin_matrix[x, 0] += self.do_subsidy(x)

    def do_subsidy(self, x):
        return self.kin_model.ff[x]


class Granddaughter(SubsidisedKin):
    def __init__(self, kin_model: KinshipModel):
        k0 = np.zeros(kin_model.n)
        super().__init__(kin_model, k0)

    def get_donor(self, x):
        return self.kin_model.daughter.get_kin_distribution(focal_age=x)


class GreatGranddaughter(SubsidisedKin):
    def __init__(self, kin_model: KinshipModel):
        k0 = np.zeros(kin_model.n)
        super().__init__(kin_model, k0)

    def get_donor(self, x):
        return self.kin_model.granddaughter.get_kin_distribution(focal_age=x)


class Mother(Kin):
    def __init__(self, kin_model: KinshipModel):
        k0 = kin_model.pp
        super().__init__(kin_model, k0)


class Grandmother(Kin):
    def __init__(self, kin_model: KinshipModel):
        k0 = np.zeros(kin_model.n)
        for i in range(kin_model.n):
            k0 += kin_model.pp[i] * kin_model.mother.get_kin_distribution(focal_age=i)
        super().__init__(kin_model, k0)


kins = {"daughter": Daughter,
        "granddaughter": Granddaughter,
        "greatgranddaughter": GreatGranddaughter,
        "mother": Mother,
        "grandmother": Grandmother}
