Requirements that will have to learn alongside to build this


1. Math & Foundations

Why: Neural networks are just math—linear algebra, calculus, probability. You need to understand it to implement forward/backward passes manually.

Concepts:

Linear Algebra

Vectors, matrices, matrix multiplication

Dot product, transpose, identity, inverse

Broadcasting (for efficient NumPy operations)

Eigenvalues/eigenvectors (optional but useful for understanding PCA)

Calculus

Derivatives of basic functions

Chain rule (critical for backpropagation)

Partial derivatives (weights & biases gradients)

Probability & Statistics

Basic probability, distributions

Mean, variance, standard deviation (normalization)

Cross-entropy intuition

Resources:

3Blue1Brown: Essence of Linear Algebra (vectors & matrix multiplication)

3Blue1Brown: Essence of Calculus (derivatives, chain rule visualizations)

StatQuest: Linear Algebra Basics, Probability & Stats Basics

2. Neural Network Theory

Why: To implement it manually, you need to know exactly what each component does.

Concepts:

Neurons, layers, architecture (input → hidden → output)

Activation functions: sigmoid, tanh, ReLU

Loss functions:

MSE for regression

Binary cross-entropy for rod detection (binary classification)

Forward propagation (how input becomes output)

Backpropagation (how errors update weights)

Gradient descent & learning rate

Resources:

3Blue1Brown: Neural Networks series (forward + backprop intuition)

StatQuest: Neural Networks from Scratch

Andrew Ng (Coursera, Week 2-3) – forward/backprop visualization

3. Image Preprocessing & Feature Engineering

Why: Raw pixels can be huge; small preprocessing improves learning for your tiny NN.

Concepts:

Grayscale conversion (average RGB)

Resizing & flattening images

Simple thresholding / edge detection

Manual “screening”: focus on regions with rods (high intensity, elongated shapes)

Optional: extract shape features (length, orientation, aspect ratio)

Resources:

OpenCV tutorials: Grayscale, Thresholding, Contours

Any basic computer vision resource (free YouTube tutorials)

4. Putting it All Together

Why: Learning theory is useless until you implement it.

Concepts to master through coding:

Initialize weights & biases manually

Forward pass: input → hidden → output

Compute loss (binary cross-entropy)

Backward pass: compute gradients (chain rule)

Update weights via gradient descent

Train on a mini dataset, visualize predictions

Resources:

Michael Nielsen: Neural Networks and Deep Learning (great for understanding backprop from scratch)

StatQuest: Gradient Descent

3Blue1Brown: Neural Network Backpropagation visualization

5. Optional Advanced Concepts (Week 3 Upgrade)

Tiny Convolutional Layer from scratch (implement simple convolution manually)

Pooling (max/average)

Data augmentation: rotation, flipping, noise

Mini-batch gradient descent

Resources:

Michael Nielsen: Convolutional Neural Networks

YouTube tutorials: CNN from scratch with NumPy

Strategy

Week 1 → Math + preprocessing + forward pass

Week 2 → Backprop + train on mini dataset

Week 3 → Improve preprocessing, try CNN manually, visualize results


*********************************************************************
AI GENERATED
