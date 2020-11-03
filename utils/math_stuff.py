import numpy as np


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def get_largest_primes(number):
    factors = prime_factors(number)

    halfway = int(np.round(len(factors)/2))
    n0 = np.prod(factors[0:halfway])
    n1 = np.prod(factors[halfway:])

    return n1, n0
