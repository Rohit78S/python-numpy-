import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
# Numpy random 
data = {
# here n = 4 number and p is 0.5 so its 50% chance
#and size = 100 so 100 random values
"binomial":random.binomial(4,0.5,100),
# for normal is loc -0 so its center is 0 and scake is 1 so its value is
# -1 to +1(68%) and -2 tp +2(95.4%), and -3 and +3(98.9%) this and for kind="kde" its gose beyond
# and size= 100 so its 100 random values
"normal": random.normal(loc=0, scale=1, size=100),
# for poisson its lam mean lambda and P(X=k) = λk.e−λ/k!, k=0,1,2....
# for lam=10 that mean Example with λ = 10:
# Mean = 10
# Std. dev. = √10 ≈ 3.16
# So most values will fall between 7 and 13, but you’ll still occasionally see 0, 20, 25, etc.
# Very large values (like 40 or 50) are possible but very rare.
"lambda": random.poisson(lam=10, size=100),
# All values in the range [0, 1) are equally likely.
# It doesn’t have a “peak” like Gaussian or Poisson — it’s flat.
#low=0.0(2 for here) (default) → minimum possible value.
#high=1.0(3 for here) (default) → maximum possible value.
#size=(2, 3) → shape of output array (2 rows, 3 columns).
# for unifrom its Spread evenly 0–1
"uni":random.uniform(2,3, size=100),
# nothing here so its we thought as loc=0 and
# its work like this most value = loc ± 3.scale = 1 ± 3x2= 1 ± 6
# That means most values fall between −5 and 7.
# About 99% of values are within roughly loc ± 3*scale.
# −∞ to ∞ for around logistic
"log": random.logistic(loc=1, scale=2, size=100),
# for exponential its Values are always ≥ 0.
#Most values are close to 0, but occasionally you get larger numbers.
# The probability decays exponentially as numbers increase.
# f(x) = 1/scale x e−x/scale, x ≥ 0
# here x is possibility u want
# Mostly near 0, decays 0 to ∞
"v":random.exponential(scale=2, size=100),
#Chi-square with df = k is the sum of squares of k standard normal variables.
# x² = z²1+z²22+....+z²k
# Here, k = df = 2, so it’s the sum of squares of 2 standard normal random numbers.
# 0 to ∞
# degrees of freedom (df)
"che": random.chisquare(df=2, size=1000),
#Values are always ≥ 0
#Shape is skewed right (like exponential, but with a peak greater than 0)
#Controlled by the scale parameter
# r = root(x²+y²)
# 0 to ∞  peak at scale, right-skewed
"ray": random.rayleigh(scale=2, size=1000),
# The mean exists only if
# 𝑎 > 1 and  𝑎 > 2
# The variance exists only if
# Range x ≥ 0
# the formula from pdf
# f(x) = a.( 1 + x)^(−(a+1)), x≥0
"dis": random.pareto(a=2, size=1000),
#remember that this zipf can give us too big value uf we want to fix it we need to use limit
#like x[x<10] but here its not define so fine remove that zipf u can see it
"zi": random.zipf(a=2, size=1000)
}
sns.displot(data, kind="kde")
plt.show()
# Multinomial Distribution
# # Formula: P(X1=x1,...,Xk=xk) = n!/(x1!...xk!) * p1^x1 * ... * pk^xk
# # Generalization of binomial for multiple categories
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)
