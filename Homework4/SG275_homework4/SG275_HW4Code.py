import numpy as np
import scipy.stats

# one coin has a probability of coming up heads of 0.2, the other 0.6
coinProbs = np.zeros(2)
coinProbs[0] = 0.2
coinProbs[1] = 0.6

# reach in and pull out a coin numTimes times
numTimes = 100

# flip it numFlips times when you do
numFlips = 2

# flips will have the number of heads we observed in 10 flips for each coin
flips = np.zeros(numTimes)
for coin in range(numTimes):
    which = np.random.binomial(1, 0.5, 1)   # randomly pick which coin (0 or 1)
    flips[coin] = np.random.binomial(numFlips, coinProbs[which], 1)  # record heads

# initialize the EM algorithm
coinProbs[0] = 0.79
coinProbs[1] = 0.51

# run the EM algorithm
for iters in range(20):
    L0 = scipy.stats.binom.pmf(flips, numFlips, coinProbs[0])
    L1 = scipy.stats.binom.pmf(flips, numFlips, coinProbs[1])

    total = L0 + L1
    total[total == 0] = 1e-12  # avoid divide by zero
    w0 = L0 / total
    w1 = L1 / total

    coinProbs[0] = np.sum(w0 * flips) / (numFlips * np.sum(w0))
    coinProbs[1] = np.sum(w1 * flips) / (numFlips * np.sum(w1))
    print(coinProbs)
