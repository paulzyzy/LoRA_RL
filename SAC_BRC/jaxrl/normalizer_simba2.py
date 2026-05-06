import numpy as np
from jaxrl.utils import Batch

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float32):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance, and count moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        ratio = batch_count / tot_count
        new_mean = self.mean + delta * ratio
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * ratio
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class RewardNormalizer(object):
    def __init__(self, num_tasks: int, discount: float = 0.99, v_max: float = 10.0):
        self.returns = np.zeros(num_tasks, dtype=np.float32)
        self.returns_max = np.zeros(num_tasks, dtype=np.float32)
        self.returns_rms = RunningMeanStd(shape=num_tasks, dtype=np.float32)
        self.discount = discount
        self.v_max = v_max

    def normalize(self, batches: Batch):
        var_denominator = np.sqrt(self.returns_rms.var + 1e-8)
        min_required_denominator = self.returns_max / self.v_max
        denominator = np.where(var_denominator > min_required_denominator, var_denominator, min_required_denominator)
        denominator = denominator[batches.task_ids]
        rewards = batches.rewards / denominator
        return Batch(observations=batches.observations, actions=batches.actions, rewards=rewards, masks=batches.masks, next_observations=batches.next_observations, task_ids=batches.task_ids)

    def update(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        done = np.logical_or(terminal, truncate)
        self.returns = self.discount * (1 - done) * self.returns + rewards
        self.returns_rms.update(self.returns[None])
        self.returns_max = np.where(self.returns_max > np.absolute(self.returns), self.returns_max, np.absolute(self.returns))
