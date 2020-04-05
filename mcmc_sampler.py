import numpy
from scipy import stats


class MCMCSampler:
    def __init__(self, target_pdf):
        self.target_pdf = target_pdf
        self.uniform = stats.uniform()

    def sample(self, x_0, sample_from_proposal_fun, n_samples, discard_first=0):
        x_prev = x_0
        for i in range(0, discard_first):
            self.sample_next(sample_from_proposal_fun, x_prev)

        result = [x_prev]
        for i in range(0, n_samples):
            x_prev = self.sample_next(sample_from_proposal_fun, x_prev)
            result.append(x_prev)
        return numpy.array(result)

    def sample_next(self, sample_from_proposal_fun, x_prev):
        proposed_x = sample_from_proposal_fun(x_prev)
        acceptance_probability = min(self.target_pdf(proposed_x) / self.target_pdf(x_prev), 1)
        if acceptance_probability >= self.uniform.rvs():
            x_prev = proposed_x
        return x_prev
