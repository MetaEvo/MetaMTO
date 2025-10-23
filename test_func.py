import numpy as np

class Problem:
    def __init__(self, dim, shift, rotate, bias):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.ub = 80
        self.lb = -80
        self.opt = self.shift
        self.optimum = self.func(self.get_optimal())

    def func(self, x):
        return 0

    def get_optimal(self):
        return self.opt

    def decode(self, x):
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        y = x - shift
        z = np.matmul(rotate, y)
        return z

class Sphere(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z = np.clip(z, a_min=-100, a_max=100)
        return np.sum(z ** 2, -1) + self.bias

class Ackley(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z = np.clip(z, a_min=-50, a_max=50)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2, -1) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z), -1) / self.dim
        return np.round(np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2), 15) + self.bias

class Griewank(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z = np.clip(z, a_min=-100, a_max=100)
        s = np.sum(z ** 2, -1)
        p = 1
        for i in range(self.dim):
            p *= np.cos(z[i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p + self.bias

class Rastrigin(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z = np.clip(z, a_min=-50, a_max=50)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, -1) + self.bias

class Rosenbrock(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z += 1
        z = np.clip(z, a_min=-50, a_max=50)
        z_ = z[1:]
        z = z[:-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias

class Weierstrass(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -0.5
        self.ub = 0.5

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z = np.clip(z, a_min=-0.5, a_max=0.5)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)), -1)
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias

class Schwefel(Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        Problem.__init__(self, dim, shift, rotate, bias)
        self.lb = -500
        self.ub = 500

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        z = np.clip(z, a_min=-500, a_max=500)
        g = z * np.sin(np.sqrt(np.abs(z)))
        return b * self.dim - np.sum(g,-1)








