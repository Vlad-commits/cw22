import numpy
from scipy import stats


class MCMCSampler:
    def __init__(self, target_pdf):
        self.target_pdf = target_pdf
        self.uniform = stats.uniform()

    def sample(self, x_0, sample_from_proposal_fun, n_samples, discard_first=0):
        x_prev = x_0
        for i in range(discard_first):
            self.sample_next(sample_from_proposal_fun, x_prev)

        result = []
        for i in range(n_samples):
            result.append(x_prev)
            x_prev = self.sample_next(sample_from_proposal_fun, x_prev)
        return numpy.array(result)

    def sample_next(self, sample_from_proposal_fun, x_prev):
        proposed_x = sample_from_proposal_fun(x_prev)
        acceptance_probability = min(self.target_pdf(proposed_x) / self.target_pdf(x_prev), 1)
        if acceptance_probability >= self.uniform.rvs():
            x_prev = proposed_x
        return x_prev
