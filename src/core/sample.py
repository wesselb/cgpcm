import numpy.random as rng
import numpy as np
from time import time

from learn import Progress


class ESS(object):
    """
    Elliptical slice sampling algorithm.

    :param log_lik: function that returns the log value of a quantity that is
                    proportional to the likelihood of the target distribution
    :param sample_prior: function that generates a sample from the prior of
                         the target distribution
    :param x_init: initial state
    """

    _min_theta = 1e-10

    def __init__(self, log_lik, sample_prior, x_init=None):
        self._log_lik = log_lik
        self._sample_prior = sample_prior
        if x_init is None:
            self._x = sample_prior()
        else:
            self._x = x_init
        self._log_lik_x = log_lik(self._x)

    def move(self, x, log_lik=None):
        """
        Move sampler to a new state.

        :param x: new state
        :param log_lik: log likelihood at new state
        """
        self._x = x
        if log_lik is None:
            self._log_lik_x = self._log_lik(x)

    def update(self, log_lik):
        """
        Update the log likelihood function.

        :param log_lik: log likelihood
        """
        self._log_lik = log_lik

    def _establish_ellipse(self):
        """
        Establish an ellipse, which is required to draw the new state.
        """
        self._y = self._sample_prior()

    def _draw_proposal(self, theta_l, theta_u):
        """
        Draw a proposal for the next state given a bracket for :math:`\\theta`.

        :param theta_l: lower bound of bracket
        :param theta_u: upper bound of bracket
        :return: proposal
        """
        self._theta = rng.uniform(theta_l, theta_u)
        self._x_proposed = np.cos(self._theta) * self._x \
                           + np.sin(self._theta) * self._y
        self._log_lik_x_proposed = self._log_lik(self._x_proposed)

    def _draw_bracket(self):
        """
        Draw a bracket for :math:`\\theta`.

        :return: bracket
        """
        theta = rng.uniform(0, 2 * np.pi)
        return theta - 2 * np.pi, theta

    def _draw(self, (theta_l, theta_u), u, attempts=1):
        """
        Draw new state given a bracket for :math:`\\theta`.

        :param theta_l: lower bound of bracket
        :param theta_u: upper bound of bracket
        :param u: slice height
        :param attempts: number of attempts
        :return: new state
        """
        self._draw_proposal(theta_l, theta_u)
        theta_violation = abs(self._theta) < self._min_theta
        if self._log_lik_x_proposed > u or theta_violation:
            # Proposal accepted
            if theta_violation:
                print 'warning: theta violation'
            return self._x_proposed, self._log_lik_x_proposed, attempts
        else:
            # Proposal rejected. Shrink bracket and try again.
            if self._theta > 0:
                return self._draw((theta_l, self._theta), u, attempts + 1)
            else:
                return self._draw((self._theta, theta_u), u, attempts + 1)

    def sample(self, num=1):
        """
        Generate samples from the target distribution.

        :param num: number of samples
        :return: samples
        """
        samples = []
        fetches_config = [{'name': 'log likelihood', 'modifier': '.0f'},
                          {'name': 'attempts', 'modifier': 'd'},
                          {'name': 'time/attempt',
                           'modifier': '.2f',
                           'unit': 'ms'}]
        progress = Progress(name='sampling using ESS',
                            iters=num,
                            fetches_config=fetches_config)
        for i in range(num):
            u = self._log_lik_x - rng.exponential(1.0)
            self._establish_ellipse()
            start_it = time()
            self._x, self._log_lik_x, attempts = self._draw(
                self._draw_bracket(), u)
            end_it = time()
            samples.append(self._x)

            # Report
            time_attempt = 1000 * (end_it - start_it) / attempts
            progress([self._log_lik_x, attempts, time_attempt])

        return samples if len(samples) > 1 else samples[0]
