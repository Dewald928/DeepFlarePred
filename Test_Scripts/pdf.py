# example of parametric probability density estimation
from matplotlib import pyplot
from numpy import asarray
from numpy import exp
from numpy import mean
from numpy import std
from numpy.random import normal
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# generate a sample
sample = normal(loc=50, scale=5, size=1000)
sample = yhat[:, 1]
# calculate parameters
sample_mean = mean(sample)
sample_std = std(sample)
print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
# define the distribution
dist = norm(sample_mean, sample_std)
# sample probabilities for a range of outcomes
values = [value for value in range(0, 2)]
probabilities = [dist.pdf(value) for value in values]
# plot the histogram and pdf
pyplot.hist(sample, bins=20, density=True)
pyplot.plot(values, probabilities)
pyplot.show()


# fit density
model = KernelDensity(bandwidth=2, kernel='gaussian')
sample = sample.reshape((len(sample), 1))
model.fit(sample)
# sample probabilities for a range of outcomes
values = asarray([value for value in range(0, 2)])
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)
# plot the histogram and pdf
pyplot.hist(sample, bins=20, density=True)
pyplot.plot(values[:], probabilities)
pyplot.show()