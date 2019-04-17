import numpy as np


class _Optimizer(object):
    name = '_Optimizer'

    def __init__(self, *args, **kwargs):
        self.iterations = 0
        self.lr = 0

    def get_update(self, params: np.ndarray, grads: np.ndarray):
        pass


class Adam(_Optimizer):
    name = 'adam'

    def __init__(self, lr: float = .001, beta1: float = .9, beta2: float = .999,
                 epsilon: float = 1e-8, decay: float = .0, **kwargs):
        super(Adam, self).__init__(**kwargs)

        self.iterations = 0
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def __next__(self):
        self.iterations += 1

    def get_update(self, params: np.ndarray, grads: np.ndarray):
        lr = self.lr * ((1. / (1. + self.deacy * self.iterations)) if self.initial_decay > 0. else 1) * \
             (np.sqrt(1. - np.power(self.beta2, 1+self.iterations)) / (1. - np.power(self.beta1, 1+self.iterations)))

        results = np.zeros(np.size(params, 0))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        for i, (p, g, m, v) in enumerate(zip(params, grads, self.ms, self.vs)):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            results[i] = p_t.reshape(params[i].shape)

        return results


Optimizers = {
    klass.name: klass for klass in _Optimizer.__subclasses__()
}
