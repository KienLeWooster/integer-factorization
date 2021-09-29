import math
import time
from collections import OrderedDict

import numpy
import sympy


def is_perfect_square(n):
    if n < 0:
        return 0
    else:
        return n == math.isqrt(n) ** 2


def sieve_prime(n):
    # This function should work for n up to around 5 * 10 ^ 8.
    # The number above is a lower bound. No upper bound has been found.
    a = [True for _ in range(2, n + 1)]
    for i in range(2, math.ceil(math.sqrt(n))):
        if a[i - 2]:
            for j in range(i ** 2, n + 1, i):
                a[j - 2] = False

    return [i + 2 for i in range(len(a)) if a[i]]


def dixon_algorithm(n):
    if n % 2 == 0:
        return [2, n / 2]
    start_time = time.time()
    # With this bound, the prime sieve in Dixon's algorithm should work for
    # n up to 10^100
    B = int(math.exp((1/2) * math.sqrt(math.log(n) * math.log(math.log(n)))))
    if B <= 10:
        B = 10
    primes = sieve_prime(B)
    pairs = OrderedDict()
    z = int(math.sqrt(n)) + 1
    # Generate relation pairs
    while len(pairs) != len(primes) + 5:
        if time.time() - start_time > 60:
            return None
        z_square_mod_n = pow(z, 2, n)
        if z_square_mod_n == 0:
            z += 1
            continue
        factor_powers = [0 for _ in range(len(primes))]
        for row in range(len(primes)):
            while z_square_mod_n % primes[row] == 0:
                factor_powers[row] += 1
                z_square_mod_n = (z_square_mod_n / primes[row])

        factor_powers = tuple(factor_powers)
        if any(factor_powers) and factor_powers not in pairs and z_square_mod_n == 1:
            pairs[factor_powers] = z
        z += 1

    if time.time() - start_time > 60:
        return None

    # Get independent rows of matrix
    factor_matrix = numpy.array([row for row in pairs])
    factor_matrix = factor_matrix % 2
    matrix = sympy.Matrix(factor_matrix.T)
    matrix = matrix.rref()
    independent_row_idx = matrix[1]
    independent_rows = factor_matrix[list(independent_row_idx)]

    del matrix

    if time.time() - start_time > 60:
        return None

    dependent_rows_idx = [i for i in range(len(pairs)) if
                          i not in independent_row_idx]

    for idx in dependent_rows_idx:
        # Find linearly dependent set of vectors
        if time.time() - start_time > 60:
            return None
        dependent_row = factor_matrix[idx]
        filler_array = numpy.zeros((len(independent_rows[0]) - len(independent_rows), len(independent_rows[0])), dtype=int)
        independent_rows = numpy.concatenate((independent_rows, filler_array), axis=0)
        sol: numpy.ndarray = numpy.linalg.lstsq(independent_rows.T, dependent_row, rcond=None)[0]
        sol = sol[:len(independent_row_idx)]
        sol = numpy.rint(sol)
        sol = sol % 2
        final_bases_idx = [independent_row_idx[i] for i in range(len(sol)) if sol[i]]
        final_bases_idx.append(idx)

        # Try to extract factor of n
        z_product = 1
        prime_product = 1
        factor_powers_list = list(pairs.keys())
        for prime_idx, i in enumerate(final_bases_idx):
            z_product *= pairs[factor_powers_list[i]]
            factor_powers = list(map(lambda x: int(x / 2), factor_powers_list[i]))
            prime_product *= math.prod(map(pow, primes, factor_powers))
        prime_product = round(prime_product)
        if not (z_product % n == (prime_product % n)) or (z_product % n == (-prime_product % n)):
            factor_1 = math.gcd(z_product + prime_product, n)
            factor_2 = math.gcd(z_product - prime_product, n)
            if factor_1 != n and factor_1 != 1:
                factors = [factor_1, n / factor_1]
                return factors
            elif factor_2 != n and factor_2 != 1:
                factors = [factor_2, n / factor_2]
                return factors


def pollard_algorithm(n):
    if n % 2 == 1:
        a = 2
    else:
        return [2, n / 2]
    start_time = time.time()
    factors = []
    # With this bound, the prime sieve in Pollard's algorithm should work for
    # n up to 10^45
    B = math.ceil(pow(n, 1 / 6))
    while True:
        if time.time() - start_time > 60:
            return None
        prime_list = sieve_prime(B)
        powers_of_a_mod_prime_power = []
        for q in prime_list:
            # Calculate a^(p^(log B/log q)) to use modular exponentiation
            powers_of_a_mod_prime_power.append(
                pow(a, pow(q, math.floor(math.log(B) / math.log(q)))))
        a_power_M_mod_n = 1
        for i in powers_of_a_mod_prime_power:
            # Calculate a^M - 1 using modular exponentiation
            a_power_M_mod_n *= i % n
        g = math.gcd(a_power_M_mod_n - 1, n)
        if 1 < g < n:
            factors.append(g)
            factors.append(n / g)
            return factors
        elif g == 1:
            B += 1
        elif g == n:
            B -= 1
            if B > prime_list[-1]:
                prime_list.pop()


def fermat_algorithm(n):
    if n % 2 == 0:
        return [2, n / 2]
    start_time = time.time()
    a = math.ceil(math.sqrt(n))
    b = a * a - n
    while not is_perfect_square(b):
        if time.time() - start_time > 60:
            return None
        b += 2 * a + 1
        a += 1
    factors = [a + math.sqrt(b), a - math.sqrt(b)]
    return factors


def trial_division(n):
    factors = []
    start_time = time.time()
    for i in range(2, n):
        if time.time() - start_time > 60:
            return None
        if n % i == 0:
            factors.append(i)
            factors.append(n / i)
            return factors


# Uncomment these lines to generate a product of two primes
# import random
# primes = sieve_prime(10000000)
# x1 = random.choice(primes)
# x2 = random.choice(primes)
# n = x1 * x2

# Uncomment this line to input a number by hand
# n = input('Enter a number: ')

# Uncomment this line to hard code the number to be factored
n = 999

# The number of times an algorithm will be run
number_of_runs = 10

print('Trial division: ')
total_time = 0
for i in range(number_of_runs):
    start_time = time.time()
    trial_division(n)
    total_time += time.time() - start_time
print(f"\tAverage time: {total_time / number_of_runs}")

print("Fermat's algorithm:")
total_time = 0
for i in range(number_of_runs):
    start_time = time.time()
    fermat_algorithm(n)
    total_time += time.time() - start_time
print(f"\tAverage time: {total_time / number_of_runs}")

print("Pollard's p - 1 algorithm")
total_time = 0
for i in range(number_of_runs):
    start_time = time.time()
    pollard_algorithm(n)
    total_time += time.time() - start_time
print(f"\tAverage time: {total_time / number_of_runs}")

print("Dixon's algorithm")
total_time = 0
for i in range(number_of_runs):
    start_time = time.time()
    dixon_algorithm(n)
    total_time += time.time() - start_time
print(f"\tAverage time: {total_time / number_of_runs}")
