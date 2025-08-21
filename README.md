# Restricted Boltzmann Machine for Iris Dataset Classification

## Overview

This project implements a **Restricted Boltzmann Machine (RBM)** from scratch to classify the famous Iris flower dataset. The implementation demonstrates how to use probabilistic neural networks for pattern recognition and classification tasks without relying on external deep learning libraries.

## About the Iris Dataset

The [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is one of the most famous datasets in machine learning, introduced by British statistician Ronald Fisher in 1936. The dataset contains:

- **150 samples** of iris flowers
- **4 features** for each sample:
  - Sepal Length (cm)
  - Sepal Width (cm) 
  - Petal Length (cm)
  - Petal Width (cm)
- **3 classes** of iris species:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

The classification task is to predict the iris species based on the four flower measurements.

## Boltzmann Machines and RBMs

### What is a Boltzmann Machine?

A **Boltzmann Machine** is a type of stochastic neural network that can learn probability distributions over binary vectors. It's named after the Boltzmann distribution from statistical mechanics, which describes the probability of a system being in a particular state based on energy.

Key characteristics:
- **Binary neurons**: Each neuron can only be in state 0 or 1
- **Stochastic activation**: Neurons activate probabilistically
- **Energy-based model**: The network state is determined by an energy function
- **Unsupervised learning**: Can learn patterns without labeled data

### Restricted Boltzmann Machine (RBM)

An **RBM** is a special type of Boltzmann machine with restrictions on connections:
- **Bipartite graph**: Only connections between visible and hidden layers
- **No intra-layer connections**: No connections within the same layer
- **Simplified training**: Makes learning more tractable than full Boltzmann machines

#### Network Architecture

```
Hidden Layer (24 neurons)   h₁  h₂  h₃ ... h₂₄
                            |   |   |      |
                            |   |   |      |
Visible Layer (15 neurons)  v₁  v₂  v₃ ... v₁₅
                           [Features: 1-12] [Classes: 13-15]
```

- **Visible layer**: 15 binary neurons (12 features + 3 classes)
- **Hidden layer**: 24 neurons (learned representations)
- **Weights**: 15×24 weight matrix connecting all visible to all hidden neurons

## Mathematical Formulation

### Data Discretization Algorithm

Since RBMs work with binary data, continuous Iris features must be discretized using a mathematical transformation:

#### Discretization Function

For each feature `f` with values `{f₁, f₂, ..., fₙ}`:

1. **Calculate maximum value**:
   ```
   f_max = max(f₁, f₂, ..., fₙ)
   ```

2. **Define thresholds**:
   ```
   threshold₁ = f_max / 3
   threshold₂ = 2 × f_max / 3
   ```

3. **Discretization mapping function**:
   ```
   discretize(fᵢ) = {
       [1, 0, 0]  if fᵢ < threshold₁     (Low)
       [0, 1, 0]  if threshold₁ ≤ fᵢ < threshold₂  (Medium)
       [0, 0, 1]  if fᵢ ≥ threshold₂     (High)
   }
   ```

#### Complete Feature Vector Transformation

For an iris sample with continuous features `[pl, pw, sl, sw]`:

```
v = [discretize(pl), discretize(pw), discretize(sl), discretize(sw), class_vector]
```

Where:
- `pl` = Petal Length
- `pw` = Petal Width  
- `sl` = Sepal Length
- `sw` = Sepal Width
- `class_vector` = [1,0,0] or [0,1,0] or [0,0,1] or [0,0,0] for inference

**Example**:
```
Input:  [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]
Output: [0,1,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0] (15-dimensional binary vector)
```

## Energy Function and Probability Distribution

### Energy Function

The energy of a joint configuration `(v,h)` is defined as:

```
E(v,h) = -∑ᵢ₌₁ⁿᵛ aᵢvᵢ - ∑ⱼ₌₁ⁿʰ bⱼhⱼ - ∑ᵢ₌₁ⁿᵛ ∑ⱼ₌₁ⁿʰ vᵢWᵢⱼhⱼ
```

Where:
- `nᵛ = 15`: number of visible units
- `nʰ = 24`: number of hidden units
- `vᵢ ∈ {0,1}`: state of visible unit i
- `hⱼ ∈ {0,1}`: state of hidden unit j
- `aᵢ`: bias of visible unit i
- `bⱼ`: bias of hidden unit j
- `Wᵢⱼ`: weight connecting visible unit i to hidden unit j

### Joint Probability Distribution

The probability of a joint configuration follows the Boltzmann distribution:

```
P(v,h) = (1/Z) × exp(-E(v,h)/T)
```

Where:
- `Z`: partition function (normalization constant)
- `T`: temperature parameter (T=1 in our implementation)

```
Z = ∑ᵥ ∑ₐ exp(-E(v,h)/T)
```

## Inference Algorithms

### Hidden Unit Conditional Probability

Given visible units, the probability of hidden unit `j` being active:

```
P(hⱼ = 1|v) = σ(bⱼ + ∑ᵢ₌₁ⁿᵛ vᵢWᵢⱼ)
```

Where `σ(x)` is the sigmoid function:
```
σ(x) = 1/(1 + exp(-x/T))
```

### Visible Unit Conditional Probability

Given hidden units, the probability of visible unit `i` being active:

```
P(vᵢ = 1|h) = σ(aᵢ + ∑ⱼ₌₁ⁿʰ hⱼWᵢⱼ)
```

### Sampling Algorithm

#### Hidden Unit Sampling:
```python
def sample_hidden(v):
    # Step 1: Calculate activation
    activation = b + v · W  # Shape: (batch_size, n_hidden)
    
    # Step 2: Apply sigmoid
    prob = σ(activation)
    
    # Step 3: Stochastic sampling
    h = (prob > random_uniform(0,1)) ? 1 : 0
    
    return h
```

Mathematical form:
```
h ~ Bernoulli(σ(b + v · W))
```

#### Visible Unit Sampling:
```python
def sample_visible(h):
    # Step 1: Calculate activation  
    activation = a + h · Wᵀ  # Shape: (batch_size, n_visible)
    
    # Step 2: Apply sigmoid
    prob = σ(activation)
    
    # Step 3: Stochastic sampling
    v = (prob > random_uniform(0,1)) ? 1 : 0
    
    return v
```

Mathematical form:
```
v ~ Bernoulli(σ(a + h · Wᵀ))
```

## Training Algorithm: Contrastive Divergence

### Contrastive Divergence Algorithm (CD-1)

The RBM is trained to maximize the log-likelihood of the training data using gradient ascent:

```
∂ln P(v)/∂Wᵢⱼ = ⟨vᵢhⱼ⟩_data - ⟨vᵢhⱼ⟩_model
```

Since computing `⟨vᵢhⱼ⟩_model` is intractable, we use Contrastive Divergence:

#### Step-by-Step CD-1 Algorithm:

**Input**: Training batch `V = {v⁽¹⁾, v⁽²⁾, ..., v⁽ᵐ⁾}`

1. **Positive Phase** - Compute data-dependent expectations:
   ```
   h⁽⁰⁾ ~ P(h|v⁽ᵈᵃᵗᵃ⁾)  # Sample hidden given data
   
   positive_W = (v⁽ᵈᵃᵗᵃ⁾)ᵀ · h⁽⁰⁾  # Outer product
   positive_a = mean(v⁽ᵈᵃᵗᵃ⁾)      # Visible bias update
   positive_b = mean(h⁽⁰⁾)          # Hidden bias update
   ```

2. **Negative Phase** - Compute model-dependent expectations:
   ```
   v⁽¹⁾ ~ P(v|h⁽⁰⁾)  # Reconstruct visible from hidden
   h⁽¹⁾ ~ P(h|v⁽¹⁾)  # Sample hidden from reconstruction
   
   negative_W = (v⁽¹⁾)ᵀ · h⁽¹⁾  # Outer product  
   negative_a = mean(v⁽¹⁾)      # Visible bias update
   negative_b = mean(h⁽¹⁾)      # Hidden bias update
   ```

3. **Parameter Updates**:
   ```
   ΔW = η × (positive_W - negative_W)
   Δa = η × (positive_a - negative_a)  
   Δb = η × (positive_b - negative_b)
   
   W ← W + ΔW
   a ← a + Δa
   b ← b + Δb
   ```

Where `η` is the learning rate.

### Mathematical Justification

The CD algorithm approximates the gradient:

```
∂ln P(v⁽ᵈᵃᵗᵃ⁾)/∂W ≈ ⟨vh⟩_data - ⟨vh⟩_reconstruction
```

This works because:
- **Positive phase**: Captures correlation between data and hidden representations
- **Negative phase**: Captures what the model currently believes about the data distribution
- **Difference**: Pushes the model toward the data distribution

### Training Loop

```python
for epoch in range(max_epochs):
    for batch in training_batches:
        # Append zeros for unknown classes during training
        batch_with_classes = append_known_classes(batch)
        
        # Run CD-1
        gradient_step(batch_with_classes, learning_rate)
        
        # Compute reconstruction error
        error = ||batch - reconstruct(batch)||²
```

## Classification Algorithm

### Inference for Unknown Samples

For a test sample with unknown class:

1. **Prepare input**: `v_test = [features, 0, 0, 0]` (unknown class)

2. **Forward pass**: 
   ```
   h = sample_hidden(v_test)
   v_reconstructed = get_visible_probabilities(h)
   ```

3. **Extract class probabilities**:
   ```
   class_probs = v_reconstructed[12:15]  # Last 3 units
   predicted_class = argmax(class_probs)
   ```

### Mathematical Form

The classification decision is:
```
ŷ = argmax P(class_k | features) 
  = argmax P(v₁₃₊ₖ = 1 | v₁:₁₂, v₁₃:₁₅ = [0,0,0])
```

Where the conditional probability is computed through the hidden layer:
```
P(v₁₃₊ₖ = 1 | v₁:₁₂) = σ(a₁₃₊ₖ + ∑ⱼ hⱼWᵢⱼ)
```

## Implementation Details

### Network Architecture Design
- **Visible units**: 15 (12 features + 3 classes)
- **Hidden units**: 24 (provides rich representations)
- **Connections**: Fully connected bipartite graph
- **Parameters**: 15×24 + 15 + 24 = 399 total parameters

### Hyperparameters
- **Learning rate**: η = 0.001
- **Batch size**: 128
- **Temperature**: T = 1
- **Epochs**: 1000
- **CD steps**: k = 1

### Performance Metrics
- **Before training**: Random performance (~33% accuracy)
- **After training**: Improved classification based on learned patterns

## Usage

```bash
python rbm.py
```

**Requirements:**
- numpy
- pandas  
- Iris dataset in `archive/Iris.csv`

**Forbidden Libraries:**
As per assignment requirements, no deep learning frameworks (PyTorch, TensorFlow, etc.) were used.

## Project Structure

```
├── restricted-boltzmann-machine.py            # Main RBM implementation
├── README.md                                  # This documentation
└── archive/
    └── Iris.csv                               # Iris dataset
```

## Technical Notes

- **Stochastic sampling**: Uses probabilistic neuron activation
- **Binary constraints**: All neurons limited to {0,1} states
- **Semi-supervised learning**: Uses class information during training
- **Generative model**: Can generate new flower patterns after training
- **Energy minimization**: Lower energy states correspond to higher probability

This implementation demonstrates the fundamental principles of energy-based models and probabilistic neural networks with detailed mathematical foundations for each component.