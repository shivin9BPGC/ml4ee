# Linear Algebra for ML - Quick Reference Cheat Sheet

## 🎯 Core Concepts & Formulas

### Vector Operations
| Operation | Formula | NumPy Code | ML Application |
|-----------|---------|------------|----------------|
| **Dot Product** | $\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$ | `np.dot(a, b)` or `a @ b` | Similarity, Neural networks |
| **L2 Norm** | $\|\vec{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$ | `np.linalg.norm(v)` | Distance, Regularization |
| **L1 Norm** | $\|\vec{v}\|_1 = \sum_{i=1}^{n} |v_i|$ | `np.sum(np.abs(v))` | Sparsity, Robust regression |
| **Cosine Similarity** | $\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$ | `np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))` | Recommendation systems |
| **Unit Vector** | $\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}$ | `v / np.linalg.norm(v)` | Normalization |

### Matrix Operations
| Operation | Formula | NumPy Code | ML Application |
|-----------|---------|------------|----------------|
| **Matrix Multiplication** | $(AB)_{ij} = \sum_{k} A_{ik}B_{kj}$ | `A @ B` or `np.matmul(A, B)` | Linear transformations |
| **Transpose** | $A^T_{ij} = A_{ji}$ | `A.T` | Data reshaping, gradients |
| **Inverse** | $AA^{-1} = I$ | `np.linalg.inv(A)` | Solving linear systems |
| **Determinant** | $\det(A)$ | `np.linalg.det(A)` | Volume scaling, singularity |
| **Trace** | $\text{tr}(A) = \sum_{i} A_{ii}$ | `np.trace(A)` | Sum of eigenvalues |

### Eigendecomposition
| Concept | Formula | NumPy Code | ML Application |
|---------|---------|------------|----------------|
| **Eigenvalue Equation** | $A\vec{v} = \lambda\vec{v}$ | `eigenvals, eigenvecs = np.linalg.eig(A)` | PCA, Spectral clustering |
| **Characteristic Polynomial** | $\det(A - \lambda I) = 0$ | Built into `np.linalg.eig()` | Finding eigenvalues |
| **Diagonalization** | $A = PDP^{-1}$ | `P @ np.diag(eigenvals) @ np.linalg.inv(P)` | Matrix powers, stability |

### Singular Value Decomposition (SVD)
| Concept | Formula | NumPy Code | ML Application |
|---------|---------|------------|----------------|
| **SVD Decomposition** | $A = U\Sigma V^T$ | `U, s, VT = np.linalg.svd(A)` | Dimensionality reduction |
| **Low-rank Approximation** | $A_k = U_k\Sigma_k V_k^T$ | `U[:,:k] @ np.diag(s[:k]) @ VT[:k,:]` | Compression, denoising |
| **Pseudoinverse** | $A^+ = V\Sigma^{-1}U^T$ | `np.linalg.pinv(A)` | Solving overdetermined systems |

---

## 🚀 Common ML Workflows

### Data Preprocessing
```python
# Standardization (Z-score normalization)
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Min-Max normalization  
def min_max_normalize(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Unit vector normalization
def unit_normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)
```

### Principal Component Analysis (PCA)
```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Select top components
    components = eigenvecs[:, :n_components]
    
    # Transform data
    X_pca = X_centered @ components
    
    return X_pca, components, eigenvals
```

### Linear Regression (Normal Equation)
```python
def linear_regression_normal_equation(X, y):
    # Add bias column
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    
    return theta
```

### Cosine Similarity Matrix
```python
def cosine_similarity_matrix(X):
    # Normalize rows to unit vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Compute pairwise cosine similarities
    similarity_matrix = X_norm @ X_norm.T
    
    return similarity_matrix
```

---

## 🔧 Essential NumPy Functions

### Array Creation & Manipulation
```python
np.array([1, 2, 3])          # Create array
np.zeros((3, 4))             # Zero matrix
np.ones((2, 3))              # Ones matrix  
np.eye(3)                    # Identity matrix
np.random.randn(3, 4)        # Random normal
np.arange(0, 10, 2)          # Range array
np.linspace(0, 1, 100)       # Linear space
```

### Array Properties & Info
```python
arr.shape                    # Dimensions
arr.size                     # Total elements
arr.ndim                     # Number of dimensions
arr.dtype                    # Data type
```

### Mathematical Operations
```python
np.sum(arr, axis=0)          # Sum along axis
np.mean(arr, axis=1)         # Mean along axis
np.std(arr)                  # Standard deviation
np.var(arr)                  # Variance
np.max(arr), np.min(arr)     # Max/min values
np.argmax(arr), np.argmin(arr) # Index of max/min
```

### Linear Algebra Specific
```python
np.linalg.norm(v, ord=2)     # Vector norms (ord=1,2,np.inf)
np.linalg.matrix_rank(A)     # Matrix rank
np.linalg.cond(A)            # Condition number
np.linalg.solve(A, b)        # Solve Ax = b
np.linalg.lstsq(A, b)        # Least squares solution
```

---

## 📊 Key ML Interpretations

### When to Use What

| Task | Method | Why |
|------|--------|-----|
| **Feature similarity** | Cosine similarity | Scale-invariant, focuses on direction |
| **Distance calculation** | L2 norm (Euclidean) | Natural geometric distance |
| **Sparse features** | L1 norm (Manhattan) | Robust to outliers |
| **Dimensionality reduction** | PCA | Preserves maximum variance |
| **Data compression** | SVD | Optimal low-rank approximation |
| **Noise reduction** | Truncated SVD | Removes low-variance components |
| **Solving linear systems** | Normal equation | Direct solution when X^T X is invertible |

### Geometric Intuitions

- **Dot product = 0**: Vectors are perpendicular (orthogonal)
- **Dot product > 0**: Vectors point in similar directions  
- **Dot product < 0**: Vectors point in opposite directions
- **High eigenvalue**: Direction of high variance in data
- **Low eigenvalue**: Direction of low variance (potential noise)
- **Singular values**: "Strength" of each principal direction

### Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|--------|----------|
| **PCA gives poor results** | Features on different scales | Standardize features first |
| **Matrix is singular** | Linearly dependent rows/columns | Use pseudoinverse or regularization |
| **Negative eigenvalues** | Matrix not positive semi-definite | Check covariance matrix calculation |
| **Poor compression** | Data is already low-rank | Check if compression is necessary |

---

## 💡 Quick Decision Guide

### Choosing Number of Components
1. **Scree plot**: Look for "elbow" in eigenvalue plot
2. **Variance threshold**: Keep components that explain 95% of variance
3. **Cross-validation**: Use validation performance as criterion
4. **Domain knowledge**: Consider interpretability requirements

### Matrix Decomposition Choice
- **PCA/Eigendecomposition**: When you need interpretable components
- **SVD**: When you need optimal low-rank approximation
- **QR decomposition**: When you need orthogonal basis (not covered in class)
- **Cholesky**: When matrix is positive definite (not covered in class)

---

## 🎓 Study Tips

### Before Exams
1. Practice computing eigenvalues/eigenvectors by hand for 2×2 matrices
2. Understand geometric interpretation of all operations
3. Know when each method is appropriate in ML context
4. Memorize key NumPy functions and their parameters

### Common Exam Questions
- Interpret principal components in terms of original features
- Calculate optimal number of PCA components
- Explain why standardization is important
- Compare dot product vs. cosine similarity
- Describe SVD applications in ML

### Programming Tips
- Always check array shapes before matrix operations
- Use `@` for matrix multiplication, `*` for element-wise
- Remember axis parameter: 0=rows, 1=columns
- Visualize results whenever possible for verification

---

*Happy learning! 🚀 Remember: Linear algebra is the language of ML - master it and you'll understand ML much more deeply.*