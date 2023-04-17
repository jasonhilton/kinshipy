import numpy as np
from abc import ABC, abstractmethod
from .projection import ProjMatSeries
from .kinship import KinshipModel

# TODO Can we write the non-timevarying model as a special case of this model
# TODO Switch to sparseness
# TODO Two Sex matrix?


def get_mother_distribution(pop_vector, fertility_rates):
    pp = pop_vector * fertility_rates
    return pp / np.sum(pp)


class KinshipModelTV(object):
    def __init__(self, proj_mat_series: ProjMatSeries, start_pop: np.ndarray):

        assert start_pop.shape[0] == proj_mat_series.n_ages
        self.proj_mat_series = proj_mat_series

        self.years = proj_mat_series.years
        self.ages = proj_mat_series.ages
        self.n_years = proj_mat_series.n_years
        self.n_ages = proj_mat_series.n_ages
        self.start_year = proj_mat_series.start_year
        self.initial_kin = KinshipModel(
            proj_mat_series.proj_mats[self.start_year]
        )

        # initial condition - population vector at t=1
        self.pop_array = np.zeros((self.n_ages, self.n_years))
        self.mother_distrib = np.zeros((self.n_ages, self.n_years))

        self.do_population_projection(start_pop)

        self.daughter = DaughterTV(self)
        self.daughter.do_projection()

        # instantiate kin here...

        # for key, KinClass in kins.items():
        #     kin_instance = KinClass(self)
        #     kin_instance.do_ageing()
        #     setattr(self, key, kin_instance)

        # age-on method?

    def do_population_projection(self, start_pop: np.ndarray):

        # assumption on size of P0
        self.pop_array[:, 0] = start_pop

        # function of t

        start_proj = self.proj_mat_series.proj_mats[self.start_year]
        self.mother_distrib[:, 0] = get_mother_distribution(start_pop,
                                                            start_proj.ff)

        for t in range(0, self.n_years - 1):
            current_year = self.start_year + t
            proj_mat = self.proj_mat_series.proj_mats[current_year]
            self.pop_array[:, t + 1] = proj_mat.dot(self.pop_array[:, t + 1])
            self.mother_distrib[:, t + 1] = get_mother_distribution(
                self.pop_array[:, t + 1], proj_mat.ff
            )

    def get_proj_mat(self, t: int):
        """
        :param t: index such that 0 = start_year
        :return: ProjMat instance corresponding to that year
        """
        return self.proj_mat_series.proj_mats[self.start_year + t]


class KinTV(ABC):

    def __init__(self, kin_model: KinshipModelTV, k0t, kx0):
        """
        :param kin_model: Instance of KinshipModelTV
        :param k0t: Initial condition for this type of kin at age 0 of focal
        :param kx0: Initial condition for this type of kin at time 0
        """
        self.kin_model = kin_model
        # kin_matrix has rows corresponding to ages of focal
        # and columns corresponding to ages of kin
        self.kin_matrix = np.zeros((kin_model.n_ages,
                                    kin_model.n_ages,
                                    kin_model.n_years))
        self.kin_matrix[:, :, 0] = kx0
        self.kin_matrix[0, :, :] = k0t

    def do_projection(self):
        for t in range(0, self.kin_model.n_years - 1):
            self.do_ageing(t)

    def do_survival(self, ka, t):
        proj_mat = self.kin_model.get_proj_mat(t)
        kb = proj_mat.UU.dot(ka)
        return kb

    def do_ageing(self, t):
        for x in range(0, self.kin_model.n_ages - 1):
            self.kin_matrix[x + 1, :, t + 1] = self.do_survival(
                self.kin_matrix[x, :, t], t
            )

    def get_kin_distribution(self, focal_age, kin_age=None, year=None):
        if kin_age is None:
            kin_age = self.kin_model.ages
        if year is None:
            year = self.kin_model.years
        return self.kin_matrix[focal_age, kin_age, year]


class SubsidisedKinTV(KinTV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_ageing(self, t):
        for x in range(0, self.kin_model.n_ages - 1):
            self.kin_matrix[x + 1, :, t + 1] = self.do_survival(
                self.kin_matrix[x, :, t], t
            )
            self.kin_matrix[x, 0, t + 1] += self.do_subsidy(
                self.get_donor(x, t), t
            )

    @abstractmethod
    def get_donor(self, age, t):
        pass

    def do_subsidy(self, donor: np.ndarray, t: int):
        proj_mat = self.kin_model.get_proj_mat(t)
        return proj_mat.ff.dot(donor)


class DaughterTV(SubsidisedKinTV):

    def __init__(self, kin_model: KinshipModelTV):
        k0t = np.zeros((kin_model.n_ages, kin_model.n_years))
        initial_kin = kin_model.initial_kin
        kx0 = initial_kin.daughter.kin_matrix
        self.kx0 = kx0
        super().__init__(kin_model, k0t, kx0)

    def get_donor(self, age: int, t: int):
        donor = np.zeros(self.kin_model.n_ages)
        donor[age] += 1
        return donor
