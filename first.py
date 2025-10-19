import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
# basic numpy 
# 3D array (2 blocks, each with 2 rows and 3 columns)
s = np.array([[[1, 2, 3], [4, 6, 5]],
              [[7, 8, 9], [10, 11, 12]]])

# Another copy of the same 3D array
s1 = np.array([[[1, 2, 3], [4, 6, 5]],
               [[7, 8, 9], [10, 11, 12]]])

# Reshape s into a new shape (2, 3, 2)
# Original shape: (2, 2, 3), reshaped to (2, 3, 2)
y = s.reshape(2, 3, 2)

# Flatten s into 1D array
y1 = s.reshape(-1)

# Number of dimensions of s
x = s.ndim

# Data type of array elements
z = s.dtype

# Split s into 1 sub-array along the first axis (axis=0 by default)
k = np.split(s, 1)

# Indices of elements where s is even
l = np.where(s % 2 == 0)

# Create a view of s (shares memory with s)
j = s.view()

# Modify the first block of the view
# Since j shares memory, original s is also affected
j[0] = 31

# Create a copy of s (independent, does not share memory)
m = s.copy()

# Concatenate s and s1 along axis=1 (rows)
d = np.concatenate((s, s1), axis=1)

# ================= Printing outputs =================
print("Concatenated array along axis=1:\n", d)
print("Flattened array:\n", y1)
print("Number of dimensions:", x)
print("Data type:", z)
print("Reshaped array (2,3,2):\n", y)
print("Sorted s along last axis:\n", np.sort(s))
print("Split array:\n", k)
print("Indices of even numbers:\n", l)
print("First 3 blocks of first dimension, first row:\n", s[0:3, 0:1])
print("View after modification:\n", j)
print("Copy (unaffected by view):\n", m)

# Iterate over all elements in 3D array and print them individually
for block in s:
    for row in block:
        for element in row:
            print(element)
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
# Numpy ufunc
def mynum(x, y):
    return x * y
mynum = np.frompyfunc(mynum, 2,1)
print(mynum([1, 2, 3, 4], [5, 6, 7, 8]))
print(type(mynum))
print(type(np.concatenate))

if type(np.add) == np.ufunc:
    print("this is")
else:
    print("wtf")

a1 = [1,2,3,4,5,5,4]
a2 = [6,7,8,9,10,11,5]

print(np.add(a1, a2))
print(np.subtract(a1, a2))
print(np.multiply(a1, a2))
print(np.divide(a1, a2))
print(np.power(a1, a2))
print(np.mod(a1, a2))
print(np.divmod(a1, a2))
print(np.remainder(a1, a2))
arr = np.array([-1, -2, 1, 2, 3, -4])
print(np.absolute(arr))
arr1 = np.trunc([-3.1666, 3.6667])
print(arr1)
arr2 = np.fix([-3.1666, 3.6667])
print(arr2)
arr3 = np.around(-3.1666, 2)
print(arr3)
arr4 = np.floor([-3.1666, 3.6667])
print(arr4)
arr5 = np.ceil([-3.1666, 3.6667])
print(arr5)
arr6 = np.arange(1, 10)
print(np.log2(arr6))

arr7 = np.frompyfunc(log, 2, 1)
print(log(100, 15))
s1 = np.sum([a1, a2])
print(s1)
s2 = np.sum([a1, a2], axis=1)
print(s2)
s3 = np.cumsum(a1)
print(s3)
s4 = np.prod([a1, a2])
print(s4)
s5 = np.prod([a1, a2], axis= 1)
print(s5)
s6 = np.cumprod(a1)
print(s6)
s7 = np.diff(a2)
print(s7)
s8 = np.diff(a2, n=2)
print(s8)
num1 = 4
num2 = 6
num3 = np.lcm(num1, num2)
print(num3)
num5 = np.lcm.reduce(a1)
print(num5)
arr9 = np.arange(1, 11)
print(np.lcm.reduce(arr9))
print(np.gcd(a1, a2))
print(np.gcd.reduce(a1))

sin1 = np.sin([np.pi/2, np.pi/3, np.pi/4])
print(sin1)
print(np.sinh(sin1))
print(np.cosh(sin1))
print(np.arcsinh(sin1))
sin2 = np.array([90, 180, 270, 360])
sin3 = np.deg2rad(sin2)
print(sin3)
sin4 = np.sin([np.pi/2, np.pi/3, np.pi/4])
sin5 = np.rad2deg(sin4)
print(sin5)
sin6 = np.arcsin(1.0)
print(sin6)
sin7 = np.array([1, -1, 0.1])
print(np.arcsin(sin7))
base = 3
prep = 4
p = np.hypot(base, prep)
print(p)
print(np.unique(a1))
print(np.union1d(a1, a2))
print(np.intersect1d(a1, a2, assume_unique=True))
print(np.setdiff1d(a1, a2, assume_unique=True))
print(np.setxor1d(a1, a2, assume_unique=True))