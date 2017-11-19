"""
Testing node evaluation in
   - linked representation
   - tensorflow
"""
import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from tfspn.piecewise import estimate_domains, estimate_bins, \
    piecewise_linear_approximation
from tfspn.tfspn import GaussianNode, PoissonNode, BernoulliNode, ProductNode, \
    SumNode, PiecewiseLinearPDFNode


data = numpy.array([[1., 2., 1.], [4., 5., 0.], [4., 5., 0.]])
print("data:", data)

n1 = GaussianNode("n1", 0, "X0", 1.0, 1.0)

assert_array_almost_equal(
    n1.eval(data), [-0.9189385332046727, -5.418938533204673, -5.418938533204673])

n2 = PoissonNode("n2", 1, "X1", 2.0)

assert_array_almost_equal(
    n2.eval(data), [-1.306852819440055, -3.321755839982319, -3.321755839982319])

n3 = BernoulliNode("n3", 2, "X2", 0.3)

assert_array_almost_equal(
    n3.eval(data), [-1.203972804325936, -0.3566749439387324, -0.3566749439387324])

p1 = ProductNode("p1", n1, n2, n3)

assert_array_almost_equal(
    p1.eval(data), [-3.429764156970663, -9.097369317125725, -9.097369317125725])


n4 = GaussianNode("n1", 0, "X0", 3.0, 1.0)

assert_array_almost_equal(
    n4.eval(data), [-2.918938533204673, -1.418938533204673, -1.418938533204673])

s1 = SumNode("s1", [0.3, 0.7], n1, n4)

assert_array_almost_equal(
    s1.eval(data), [-1.848479922904004, -1.767794565136819, -1.767794565136819])


domains = estimate_domains(data, ["continuous"])

print("domains:", domains)

bins = estimate_bins(data[:, 0], "continuous", domains[0])

print("bins: ", bins)

x_range, y_range = piecewise_linear_approximation(data[:, 0], bins, family="continuous")

print(x_range, y_range)

n5 = PiecewiseLinearPDFNode("pwl1", 0, "X0", domains[0], x_range, y_range)

print(numpy.exp(n5.eval(data)))

print(n5.eval(data))


family = "discrete"
domains = estimate_domains(data, [family])

print("domains:", domains)

bins = estimate_bins(data[:, 0], family, domains[0])

print("bins: ", bins)

x_range, y_range = piecewise_linear_approximation(data[:, 0], bins, family=family)

print(x_range, y_range)

n5 = PiecewiseLinearPDFNode("pwl1", 0, "X0", domains[0], x_range, y_range)

print(numpy.exp(n5.eval(data)))

print(n5.eval(data))
