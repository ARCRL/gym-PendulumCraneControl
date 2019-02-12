import numpy as np
import gym

class Continuous(gym.Space):
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
        gym.Space.__init__(self, (), np.float64)

    def sample(self):
        return (gym.spaces.np_random.rand()-0.5)*self.n

    def contains(self, x):
        #print(x)
        #print(isinstance(x, (np.generic, np.ndarray)), x.dtype.kind in np.typecodes['Float'], x.shape == ())
        if isinstance(x, float):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['Float'] and x.shape == ()):
            as_int = float(x)
        else:
            return False
        return as_int >= -self.n/2 and as_int <= self.n/2

    def __repr__(self):
        return "Continuous(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
