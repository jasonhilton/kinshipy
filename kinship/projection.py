import numpy as np
import pandas as pd

# TODO Allow ProjMat to inherit from ndarray?
# TODO Two Sex matrix?
# TODO Document


def get_proj_mat(year, fert_df, mort_df, sr=100/205):
    """
    Assumptions on dataframes
    """
    # check that fert_df and mort_df have same years - take subset. 
    
    ff = fert_df.loc[fert_df.Year == year, :]["Fert"].to_numpy()
    qq = mort_df.loc[mort_df.Year == year, :]["Mort_F"].to_numpy()  # ugly
    ss = 1 - qq
    ss[ss < 0] = 0
    return ProjMat(ss, ff * sr)

# Proj mats - should this be an n-d-array?


# provide indexing functions so that [,] can be used?
class ProjMatSeries(object):
    def __init__(self, fert_df, mort_df):
        """
        assume fert has Age, Year, Fert
        assume mort has Age, Year, Mort
        rates per person not per thousand
        """
        # sort out NAs first
        # or warn? replace pop
        # checks for correctness of data frame...
        fert_years = fert_df.dropna().Year.unique()
        fert_years.sort()  # inplace, annoyingly
        mort_years = mort_df.dropna().Year.unique()
        mort_years.sort()
        self.years = fert_years[pd.Series(fert_years).isin(mort_years)]
        mort_ages = mort_df.Age.unique()
        fert_ages = fert_df.Age.unique()
        self.ages = fert_ages[pd.Series(fert_ages).isin(mort_ages)]
        self.n_years = self.years.shape[0]
        self.n_ages = self.ages.shape[0]
        fert_df = fert_df[fert_df.Year.isin(self.years)]
        self.fert_df = fert_df[fert_df.Age.isin(self.ages)]
        mort_df = mort_df[mort_df.Year.isin(self.years)]
        self.mort_df = mort_df[mort_df.Age.isin(self.ages)]
        self.proj_mats = {year: self._construct_proj_mat(year) for year in self.years}
        self.start_year = np.min(self.years)

    def _construct_proj_mat(self, year, sr=100 / 205):
        """
        Assumptions on dataframes
        """
        # check that fert_df and mort_df have same years - take subset.

        ff = self.fert_df.loc[self.fert_df.Year == year, :]["Fert"].to_numpy()
        qq = self.mort_df.loc[self.mort_df.Year == year, :]["Mort_F"].to_numpy()  # ugly
        ss = 1 - qq
        ss[ss < 0] = 0
        return ProjMat(ss, ff * sr)


class ProjMat(object):

    def __init__(self, ss, ff):
        """
        Make Projection matrix (R in caswell's notation)
        :param ss: numpy 1-d array of survival values
        :param ff: numpy 1-d array of fertility rates
        :returns: projection matrix corresponding to ss as a 2-d numpy array
        """

        assert type(ss) == np.ndarray
        assert type(ff) == np.ndarray
        assert ff.shape[0] == ss.shape[0]
        self.nn = ss.shape[0]
        self.UU = make_u_matrix(ss)
        self.RR = self.UU.copy()
        self.RR[0, :] = ff
        self.ff = ff

        eigvals, eigvecs = np.linalg.eig(self.RR)
        max_ind = np.argmax(eigvals)
        ww = np.abs(np.real(eigvecs[:, max_ind]))
        self.stable_age_dist = ww / ww.sum()
        pp = self.stable_age_dist * ff
        self.mother_dist = pp / sum(pp)

    def dot(self, *args):
        self.RR.dot(*args)


def make_u_matrix(ss):
    """
    Make survival matrix (U in caswell's notation)
    :param ss: vector of survival values. Can be list or array.

    :returns: survival matrix corresponding to ss as a 2-d numpy array
    """
    nn = len(ss)
    mat1 = np.diag(ss[0:(nn - 1)])
    mat2 = np.vstack([np.zeros((1, nn - 1)), mat1])
    end_vec = np.zeros((nn, 1))
    end_vec[-1] = ss[-1]
    mat3 = np.hstack([mat2, end_vec])
    return mat3
