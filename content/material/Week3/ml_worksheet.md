# Machine Learning Fundamentals Worksheet
**Time Allocation: 60 minutes**

---

## Part 1: Overview of Machine Learning Process (15 minutes)

### Exercise 1.1: The ML Pipeline (5 minutes)
Below is a scrambled list of steps in the machine learning process. **Number them in the correct order (1-7):**

- __ Deploy and monitor the model
- __ Collect and prepare data
- __ Define the problem and goals
- __ Evaluate model performance
- __ Choose and train a model
- __ Split data into training/validation/test sets
- __ Feature engineering and selection

### Exercise 1.2: Problem Classification (5 minutes)
**Match each real-world problem to its ML task type:**

| Problem | ML Task Type |
|---------|--------------|
| Predicting house prices | A. Classification |
| Detecting spam emails | B. Regression |
| Recommending movies | C. Clustering |
| Grouping customers by behavior | D. Recommendation |
| Predicting stock prices | |
| Diagnosing medical conditions | |

### Exercise 1.3: Scenario Analysis (5 minutes)
**For each scenario, identify which stage of the ML pipeline has a problem:**

1. Your model predicts house prices perfectly on training data but terribly on new houses.
   - Problem stage: ________________

2. Your spam detector works great for English emails but fails on other languages.
   - Problem stage: ________________

3. Your recommendation system is too slow to use in production.
   - Problem stage: ________________

---

## Part 2: K-Nearest Neighbors Algorithm (30 minutes)

### Understanding K-NN Conceptually (10 minutes)

### Exercise 2.1: Manual Classification (5 minutes)
Look at this 2D plot where X marks represent "cats" and O marks represent "dogs":

```
    |
  4 |   X     O
    |
  3 |     ?   X
    |
  2 |   O     X
    |
  1 |     X
    |
  0 +---+---+---+---
    0   1   2   3   4
```

The `?` at position (2, 3) is a new data point we want to classify.

**Questions:**
1. Using K=1 (1-nearest neighbor), what would `?` be classified as? ________________
2. Using K=3 (3-nearest neighbors), what would `?` be classified as? ________________
3. Which K value do you think would be more reliable and why?

### Exercise 2.2: Distance Intuition (5 minutes)
Calculate the Euclidean distance from point `?` at (2, 3) to each neighbor:

- Distance to X at (1, 4): √[(2-1)² + (3-4)²] = √[1 + 1] = √2 ≈ 1.41
- Distance to O at (3, 4): √[___ + ___] = ________________
- Distance to X at (3, 3): √[___ + ___] = ________________
- Distance to X at (2, 2): √[___ + ___] = ________________

Rank the neighbors from closest to farthest: ________________

### Implementing K-NN Step by Step (20 minutes)

Now let's code K-NN from scratch! Work through each step below.

### Exercise 2.3: Distance Function (4 minutes)

```python
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: tuple or array of (x, y)
        point2: tuple or array of (x, y)
    
    Returns:
        float: Euclidean distance
    """
    # TODO: Implement Euclidean distance formula
    # Hint: √[(x1-x2)² + (y1-y2)²]
    
    pass

# Test your function
print(euclidean_distance((0, 0), (3, 4)))  # Should print 5.0
print(euclidean_distance((1, 2), (1, 2)))  # Should print 0.0
```

### Exercise 2.4: Find K Nearest Neighbors (6 minutes)

```python
def find_k_nearest(test_point, training_data, k):
    """
    Find the k nearest neighbors to a test point.
    
    Args:
        test_point: tuple of (x, y)
        training_data: list of tuples [(x, y, label), ...]
        k: number of neighbors to find
    
    Returns:
        list: k nearest neighbors with their labels
    """
    distances = []
    
    # TODO: Calculate distance from test_point to each training point
    # Hint: Use your euclidean_distance function
    for x, y, label in training_data:
        # Calculate distance and store with label
        pass
    
    # TODO: Sort by distance and return k nearest
    # Hint: Use sorted() with a lambda function
    
    pass

# Test data
training_data = [
    (1, 1, 'cat'), (2, 1, 'cat'), (1, 2, 'cat'),
    (4, 4, 'dog'), (5, 4, 'dog'), (4, 5, 'dog')
]

test_point = (2.5, 2.5)
neighbors = find_k_nearest(test_point, training_data, k=3)
print(neighbors)  # Should show 3 nearest neighbors
```

### Exercise 2.5: Majority Voting (5 minutes)

```python
def predict_class(neighbors):
    """
    Predict class based on majority vote of neighbors.
    
    Args:
        neighbors: list of tuples [(x, y, label, distance), ...]
    
    Returns:
        string: predicted class label
    """
    # TODO: Count votes for each class
    # Hint: Use a dictionary to count labels
    
    pass

def knn_classifier(test_point, training_data, k):
    """
    Complete K-NN classifier.
    """
    neighbors = find_k_nearest(test_point, training_data, k)
    prediction = predict_class(neighbors)
    return prediction, neighbors

# Test your classifier
prediction, neighbors = knn_classifier((2.5, 2.5), training_data, k=3)
print(f"Prediction: {prediction}")
print(f"Based on neighbors: {neighbors}")
```

### Exercise 2.6: Experimenting with K (5 minutes)

```python
# Let's test different K values
test_points = [(2, 2), (3, 3), (1.5, 4)]

for point in test_points:
    print(f"\nPredicting for point {point}:")
    for k in [1, 3, 5]:
        pred, _ = knn_classifier(point, training_data, k)
        print(f"  K={k}: {pred}")

# Questions to think about:
# 1. Which K value seems most stable?
# 2. What happens when K is too small? Too large?
# 3. Why might K=1 be unreliable?
```

---

## Part 3: Measuring and Understanding Error (12 minutes)

### Exercise 3.1: Accuracy Calculation (4 minutes)

```python
def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculate classification accuracy.
    
    Args:
        true_labels: list of actual labels
        predicted_labels: list of predicted labels
    
    Returns:
        float: accuracy as a percentage
    """
    # TODO: Calculate what percentage of predictions are correct
    
    pass

# Test your function
true = ['cat', 'dog', 'cat', 'dog', 'cat']
pred = ['cat', 'dog', 'dog', 'dog', 'cat']
accuracy = calculate_accuracy(true, pred)
print(f"Accuracy: {accuracy}%")  # Should print 80.0%
```

### Exercise 3.2: Error Analysis with Different K Values (5 minutes)

```python
# Extended test dataset
test_data = [
    ((1.5, 1.5), 'cat'), ((2.2, 1.8), 'cat'), ((0.8, 2.1), 'cat'),
    ((4.1, 3.9), 'dog'), ((4.8, 4.2), 'dog'), ((3.9, 4.8), 'dog')
]

# Test different K values
k_values = [1, 3, 5]
accuracies = []

for k in k_values:
    predictions = []
    true_labels = []
    
    for (x, y), true_label in test_data:
        pred, _ = knn_classifier((x, y), training_data, k)
        predictions.append(pred)
        true_labels.append(true_label)
    
    accuracy = calculate_accuracy(true_labels, predictions)
    accuracies.append(accuracy)
    print(f"K={k}: Accuracy = {accuracy}%")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, 'bo-')
plt.xlabel('K Value')
plt.ylabel('Accuracy (%)')
plt.title('K-NN Accuracy vs K Value')
plt.grid(True)
plt.show()
```

### Exercise 3.3: Understanding Error Types (3 minutes)

**Scenario Analysis:**

1. **High Bias (Underfitting)**: Your K-NN uses K=100 on a dataset of 120 points.
   - What problem might this cause? ________________
   - How would you fix it? ________________

2. **High Variance (Overfitting)**: Your K-NN uses K=1 and memorizes every training example.
   - What problem might this cause? ________________
   - How would you fix it? ________________

3. **Just Right**: Your K-NN uses K=5 and generalizes well to new data.
   - Why might this be a good choice? ________________

---

## Part 4: Integration and Reflection (3 minutes)

### Exercise 4.1: Connecting the Concepts

**Quick Quiz:**

1. In which stage of the ML process do we choose the value of K?
   - A) Data collection  B) Model training  C) Model evaluation  D) Feature engineering

2. If your K-NN has 90% accuracy on training data but 60% on test data, what's likely happening?
   - A) Underfitting  B) Overfitting  C) Perfect fit  D) Data corruption

3. For a dataset with 1000 points evenly split between 2 classes, what's a reasonable starting value for K?
   - A) K=1  B) K=5  C) K=500  D) K=999

### Exercise 4.2: Real-World Applications

**Think of a real-world problem where K-NN might be useful and answer:**

1. Problem: ________________
2. What would your features (x, y coordinates) represent? ________________
3. What would you be classifying? ________________
4. What K value would you start with? ________________

---

## Answer Key

### Part 1 Answers:
**1.1:** 3, 2, 1, 6, 5, 4, 7
**1.2:** House prices-B, Spam-A, Movies-D, Customers-C, Stocks-B, Medical-A
**1.3:** 1-Evaluation/overfitting, 2-Data preparation, 3-Deployment

### Part 2 Sample Solutions:
```python
# Exercise 2.3
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Exercise 2.4
def find_k_nearest(test_point, training_data, k):
    distances = []
    for x, y, label in training_data:
        dist = euclidean_distance(test_point, (x, y))
        distances.append((x, y, label, dist))
    
    distances.sort(key=lambda x: x[3])
    return distances[:k]

# Exercise 2.5
def predict_class(neighbors):
    votes = {}
    for _, _, label, _ in neighbors:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get)
```

### Part 3 Sample Solution:
```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    return (correct / len(true_labels)) * 100
```

### Part 4 Answers:
**4.1:** 1-C, 2-B, 3-B