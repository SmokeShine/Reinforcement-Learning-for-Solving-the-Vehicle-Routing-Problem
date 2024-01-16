# Reinforcement Learning for Solving the Vehicle Routing Problem

This repository contains pytorch implementation for solving VRP using Reinforcement Learning[1]

# Reward Function

1. Advantage = total distance of predicted route - critic value
1. Actor = advantage* log probability
1. Critic = advantage^2  

# Model Details

## Actor

### Encoder

CNN Embedding

### Decoder

Single LSTM Layer

## Critic

1. 2 Dense Layers

## Optimizer

1. SGD
1. learning rate 1.0
1. L2 Gradient Clipping 2.0
1. Batch Size 128

## Hardware

1. OS: macOS 14.0 23A344 arm64
1. Host: MacBookPro17,1
1. CPU: Apple M1
1. GPU: Apple M1
1. Memory: 16384MiB

# Links

[1] Nazari, M. et al.: Reinforcement Learning for Solving the Vehicle Routing Problem.
