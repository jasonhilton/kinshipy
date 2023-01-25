import numpy as np

# TODO Allow ProjMat to inherit from ndarray?
# TODO Two Sex matrix?
# TODO Document


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
        ww = np.real(eigvecs[:, max_ind])
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
