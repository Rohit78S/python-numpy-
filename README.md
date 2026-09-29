# NumPy Complete Tutorial

A comprehensive guide to NumPy, covering array operations, random distributions, mathematical functions, and more. This repository includes practical code examples and detailed explanations for beginners and intermediate users.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Features](#features)
- [Topics Covered](#topics-covered)
- [Getting Started](#getting-started)
- [Code Structure](#code-structure)
- [Key Concepts](#key-concepts)
- [Examples](#examples)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a complete, hands-on introduction to NumPy—Python's fundamental package for numerical computing. Whether you're new to data science or looking to deepen your NumPy skills, this tutorial covers essential concepts with practical examples.

NumPy is used for:
- Working with multi-dimensional arrays and matrices
- Performing mathematical and statistical operations
- Data manipulation and analysis
- Building foundations for machine learning and scientific computing

## Installation

### Prerequisites
- Python 3.7 or higher

### Install Required Libraries

```bash
pip install numpy matplotlib seaborn
```

Or install using conda:

```bash
conda install numpy matplotlib seaborn
```

## Features

- **Beginner-friendly**: Detailed comments explaining each operation
- **Comprehensive coverage**: From basic arrays to advanced mathematical functions
- **Well-organized**: Topics grouped logically with clear sections
- **Interactive**: Run examples directly and modify them to experiment
- **Visualizations**: Includes matplotlib and seaborn for distribution plotting

## Topics Covered

### 1. Basic Array Operations
- Creating multi-dimensional arrays (1D, 2D, 3D)
- Array properties (shape, dimensions, data type)
- Reshaping and flattening arrays
- Concatenation, splitting, and sorting
- Indexing and slicing
- Views vs. Copies (memory management)

### 2. Random Number Generation
- **Binomial distribution**: Modeling number of successes in trials
- **Normal (Gaussian) distribution**: Bell curve, most common in statistics
- **Poisson distribution**: Modeling rare event occurrences
- **Uniform distribution**: Equal probability across range
- **Logistic distribution**: S-shaped cumulative distribution
- **Exponential distribution**: Modeling time between events
- **Chi-square distribution**: Hypothesis testing
- **Rayleigh distribution**: Magnitude of 2D normal vectors
- **Pareto distribution**: Power-law or "80/20" distributions
- **Zipf distribution**: Rank-based distributions
- **Multinomial distribution**: Extension of binomial to multiple categories

### 3. Universal Functions (ufuncs)
- Creating custom ufuncs
- Arithmetic operations: add, subtract, multiply, divide, power
- Modulo and remainder operations
- Absolute value, truncation, rounding, floor, ceiling

### 4. Logarithmic and Exponential Functions
- Base-2 logarithms (log2)
- Custom base logarithms
- Exponential operations

### 5. Aggregation and Reduction
- Summation and cumulative sums
- Products and cumulative products
- Differences between consecutive elements
- LCM (Least Common Multiple) and GCD (Greatest Common Divisor)

### 6. Trigonometric and Hyperbolic Functions
- Sine, cosine, tangent functions
- Hyperbolic functions (sinh, cosh)
- Inverse trigonometric functions (arcsin, arccos)
- Degree to radian conversion
- Pythagorean theorem calculations

### 7. Set Operations
- Finding unique elements
- Union, intersection, difference
- Symmetric difference (exclusive or)

## Getting Started

### Running the Tutorial

```bash
python numpy_tutorial.py
```

This will execute all demonstrations and display results in your terminal.

### Exploring Specific Sections

Each section is clearly marked with comments. You can:
1. Read through the commented code
2. Run specific sections individually
3. Modify values and observe changes
4. Experiment with different array sizes and parameters

## Code Structure

```
numpy_tutorial.py
├── Basic Array Operations
│   ├── 3D Array Creation
│   ├── Reshaping & Flattening
│   ├── Concatenation & Splitting
│   └── Views vs Copies
│
├── Random Number Generation
│   ├── Various distributions
│   └── Visualization with seaborn
│
├── Universal Functions (ufuncs)
│   ├── Custom ufunc creation
│   └── Arithmetic operations
│
├── Rounding & Truncation
│   ├── Floor, ceiling, rounding
│   └── Truncation methods
│
├── Aggregation Functions
│   ├── Sum, product operations
│   ├── LCM, GCD calculations
│   └── Cumulative operations
│
├── Trigonometric Functions
│   ├── Basic trig functions
│   ├── Hyperbolic functions
│   └── Degree-radian conversion
│
└── Set Operations
    ├── Unique elements
    ├── Union, intersection
    └── Difference operations
```

## Key Concepts

### Arrays and Shapes
Arrays are the foundation of NumPy. They can have multiple dimensions:
- **1D array**: Like a list or vector
- **2D array**: Like a matrix or spreadsheet
- **3D array**: Like stacked matrices (blocks)

### Views vs Copies
```python
j = s.view()    # Shares memory - changes affect original
m = s.copy()    # Independent - changes don't affect original
```

### Universal Functions (ufuncs)
NumPy functions that operate element-wise on arrays for speed and efficiency.

### Broadcasting
NumPy automatically extends operations across arrays of different shapes when possible.

## Examples

### Creating and Manipulating Arrays
```python
import numpy as np

# Create a 3D array
arr = np.array([[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]])

# Reshape to new dimensions
reshaped = arr.reshape(2, 3, 2)

# Flatten to 1D
flattened = arr.reshape(-1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Find even numbers
indices = np.where(arr % 2 == 0)
```

### Working with Random Numbers
```python
# Generate 100 random values from normal distribution
data = np.random.normal(loc=0, scale=1, size=100)

# Poisson distribution (modeling rare events)
events = np.random.poisson(lam=10, size=100)

# Create visualizations
import matplotlib.pyplot as plt
plt.hist(data, bins=20)
plt.show()
```

### Mathematical Operations
```python
a1 = [1, 2, 3, 4, 5]
a2 = [6, 7, 8, 9, 10]

# Element-wise operations
np.add(a1, a2)          # [7, 9, 11, 13, 15]
np.multiply(a1, a2)     # [6, 14, 24, 36, 50]

# Aggregation
np.sum(a1)              # 15
np.prod(a1)             # 120
np.cumsum(a1)           # [1, 3, 6, 10, 15]
```

### Set Operations
```python
a1 = [1, 2, 3, 4, 5, 5, 4]
a2 = [1, 2, 1, 2, 3, 4]

np.unique(a1)                    # [1, 2, 3, 4, 5]
np.union1d(a1, a2)               # Combine all unique elements
np.intersect1d(a1, a2)           # Common elements
np.setdiff1d(a1, a2)             # Elements in a1 but not a2
```

## Resources

### Official Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy API Reference](https://numpy.org/doc/stable/reference/)

### Learning Resources
- [NumPy Official Tutorials](https://numpy.org/learn/)
- [Data Science Handbook - NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html)
- [Real Python NumPy Guide](https://realpython.com/numpy-tutorial/)

### Related Libraries
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing
- **Scikit-learn**: Machine learning

## Contributing

Contributions are welcome! If you find issues or want to add content:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact & Questions

If you have questions or suggestions:
- Open an issue on GitHub
- Submit a discussion topic
- Reach out with feedback

## Acknowledgments

This tutorial covers concepts from NumPy's official documentation and best practices from the scientific Python community.

---

**Happy Learning! 🚀** Start with the basics and gradually explore more advanced topics. NumPy is a powerful tool that will enhance your data science and scientific computing journey!
