# Training-Transparent Neural Networks with Learned Interpretable Features

## Abstract

This paper introduces a neural network architecture that maintains interpretability during training while achieving competitive or superior performance on specific sequential prediction tasks. The Pattern Predictive Network (PPN) uses learned pattern dictionaries and prediction-error based routing to create transparent information flow throughout the network. In tests against standard neural architectures, the base PPN achieved a 76% reduction in prediction error on chaotic sequence data while maintaining interpretable internal representations. A hierarchical variant demonstrated a 32% improvement on pattern memory tasks, though with performance trade-offs that were clearly observable through the architecture's inherent transparency. These results suggest that the traditional trade-off between neural network performance and interpretability may not be fundamental, and demonstrate the feasibility of designing neural networks that are transparent by construction rather than requiring post-hoc interpretation methods.

## 1. Introduction

Interpretability is a core concern in AI research. Neural networks, the basic structure underlying almost all recent AI breakthroughs, are typically considered "black boxes" in which it is difficult or impossible to understand the underlying decision-making process. As models grow larger and more capable, the complexity of understanding their decision making increases. The implications range from large-scale concerns about the safety and possible underlying intentions of the most capable AIs to concerns about life-altering administrative decisions being made in areas such as law, healthcare, and insurance without any facility for affected individuals to query the underlying reasoning. However, there has historically been a trade-off between transparency and performance in AI development.

This work presents a novel neural network architecture inspired by predictive processing theory. The Pattern Predictive Network (PPN) architecture demonstrates substantial performance improvements over standard neural net architecture across specific data types while learning interpretable features at training time and maintaining transparent information flow throughout the network.

Two key features distinguish the PPN architecture from standard neural networks:

1. **Pattern Dictionary**
   - Learns interpretable patterns at each layer
   - Enables direct visualization of learned features
   - Tracks and exposes pattern usage statistics

2. **Prediction Error Based Routing**
   - Each layer predicts the next layer's compression of its information
   - If prediction is accurate, information routes to the pre-output layer rather than the next layer
   - Routing decisions are modulated by uncertainty
   - Key intuition: if information doesn't surprise the next layer, why tell it?

## 2. Methods

### 2.1 Implementation Environment

All experiments were conducted using Python 3.10.16. Network architectures were implemented using PyTorch, with analysis and visualization performed using NumPy and Matplotlib.

### 2.2 Data Generation

Three distinct types of sequential data were used to evaluate the architectures:

#### Lorenz Data

- Time series data generated from the Lorenz attractor system
- Input sequences of length 20 containing continuous values
- Target values representing the next point in the sequence
- Exhibits chaotic behavior and sensitivity to initial conditions

#### Pattern Memory Data

- Sequences with future values dependent on earlier patterns
- Input sequences of length 20 containing binary values (-1 or 1)
- Target values corresponding to patterns from earlier in the sequence
- Tests long-term dependency learning

#### Language Data

- Synthetic language-like sequences using a simple grammar
- Input sequences of length 20 representing characters (0 for space, 1-26 for letters)
- Target values predicting the next character
- Generated using patterns like "the cat sat on the mat"

### 2.3 Network Architectures

Two variants of the Pattern Predictive Network were tested against a standard feedforward neural network:

#### Base PPN Architecture

- Pattern-based compression at each layer using learned dictionaries
- Predictive connections between layers
- Dynamic routing based on prediction accuracy
- Confidence-weighted contributions to final output

#### Hierarchical PPN Architecture

- Extended base architecture with multi-level pattern hierarchies
- Implemented compression factor between levels
- Enhanced pattern recognition capabilities
- Maintained transparent information flow

#### Standard Neural Network (Control)

- Feedforward architecture with equivalent layer dimensions
- ReLU activation functions
- Standard backpropagation training

### 2.4 Experimental Design

- Five independent trials with different random seeds
- 100 training epochs per trial
- Batch size of 32
- 80/20 train/test split
- Evaluation metrics tracked every 5 epochs
- Performance compared using independent t-tests on final test losses

### 2.5 Performance Metrics

**Primary metrics:**

- Mean squared error on test set
- Pattern entropy across layers
- Layer-specific confidence values
- Inter-layer prediction accuracy

**Additional tracking:**

- Pattern usage distribution
- Flow magnitudes between layers
- Prediction error propagation
- Layer-wise contribution to final predictions

## 3. Results

### 3.1 Base Architecture Performance

The Pattern Predictive Network (PPN) demonstrated markedly different performance characteristics across the three data types tested. For the Lorenz attractor data, the PPN significantly outperformed the standard neural network, achieving a mean test loss of 0.00498 (SD = 0.0021) compared to 0.0207 (SD = 0.0044) for the standard network (t = 6.447, p < 0.001). This represents a 76% reduction in prediction error for chaotic sequences.

For pattern memory sequences, the base PPN showed marginal improvement (mean loss 1.694, SD = 0.100) compared to the standard network (mean loss 1.809, SD = 0.140), though this difference was not statistically significant (t = 1.329, p = 0.221). Language data testing showed nearly identical performance between architectures (PPN mean loss 3.240, SD = 0.329; standard mean loss 3.240, SD = 0.339; t = -0.001, p = 0.999).

### 3.2 Pattern Analysis

The transparency of the architecture enabled detailed analysis of pattern formation and usage. Figures 1 and 2 show pattern projections onto the Lorenz attractor for layers 0 and 1 respectively. Different patterns became selectively responsive to specific regions of the attractor's state space, demonstrating specialized feature detection.
![Figure 1](/images/Fig1.png)
[Figure 1: Layer 0 pattern projections onto the Lorenz attractor. Each subplot shows the activation strength of a different learned pattern across the attractor's state space, demonstrating specialization to different dynamical regions.]
![Figure 2](/images/Fig2.png)
[Figure 2: Layer 1 pattern projections onto the Lorenz attractor, showing higher-level pattern specialization. Note the different activation distributions compared to Layer 0, suggesting hierarchical feature extraction.]

Pattern usage evolved systematically during sequence processing, as shown in Figures 3 and 4. These visualizations demonstrate dynamic transitions in pattern activation corresponding to different phases of the input sequences.
![Figure 3](/images/Fig3.png)
[Figure 3: Pattern usage evolution and information flow in Layer 0. Upper panel shows pattern activation over time; lower panel shows routing decisions between direct output contribution and continued processing.]
![Figure 4](/images/Fig4.png)
[Figure 4: Pattern usage evolution and information flow in Layer 1, demonstrating different pattern specialization and routing dynamics compared to Layer 0.]

Pattern correlation analysis (Figure 5) revealed largely independent pattern development, particularly in lower layers. The pattern usage distribution (Figures 6 and 7) showed clear clustering of activations, suggesting specialization of patterns for distinct sequence features.
![Figure 5](/images/Fig5.png)
[Figure 5: Pattern correlations across layers, showing independence of learned patterns. Red indicates positive correlation, blue indicates negative correlation.]
![Figure 6](/images/Fig6.png)
![Figure 7](/images/Fig7.png)
[Figures 6 and 7: Pattern usage by cluster for Layers 0 and 1, demonstrating specialized pattern activation for different input sequence types.]

### 3.3 Hierarchical Extension Results

The hierarchical variant of the PPN architecture showed a dramatically different performance profile. For pattern memory sequences, it achieved significantly better performance (mean loss 1.227, SD = 0.072) compared to the standard network (mean loss 1.809, SD = 0.140; t = 7.380, p < 0.0001), representing a 32% reduction in prediction error.

However, this architectural modification severely impacted performance on the Lorenz data (mean loss 5.725, SD = 1.118) compared to both the standard network (mean loss 0.021, SD = 0.004; t = -10.207, p < 0.00001) and the base PPN. Language modeling showed no significant improvement (mean loss 3.661, SD = 0.649 vs 3.240, SD = 0.339; t = -1.150, p = 0.283).

For language data specifically, pattern activations at word boundaries (Figure 8) and character-pattern associations (Figure 9) revealed limited structural organization, suggesting the current architecture may be suboptimal for discrete symbolic sequences.

![Figure 8](/images/Fig8.png)
[Figure 8: Layer 0 pattern activation at word boundaries, showing limited differentiation between boundary and non-boundary positions.]

![Figure 9](/images/Fig9.png)
[Figure 9: Layer 0 pattern-character associations, demonstrating weak specialization for specific character types.]

## 4. Discussion

The primary finding of this investigation is a neural network architecture that allows direct observation of internal representations during training while achieving competitive or superior performance on specific tasks. The pattern predictive network's transparency is not achieved through post-hoc interpretation methods, but is inherent in its architecture and observable throughout the training process.

The base architecture demonstrated stable training behavior across multiple data types. For the Lorenz attractor system, it achieved a 76% reduction in prediction error compared to standard architectures while maintaining interpretable internal representations. Pattern projections onto the Lorenz attractor (Figures 1-2) reveal that the network develops specialized detectors for different regions of the attractor's state space, providing direct insight into its information processing. This transparency enabled verification that the network was learning meaningful features of the underlying dynamical system rather than exploiting spurious correlations.

The hierarchical variant's results highlight both the potential and current limitations of this approach. Its 32% reduction in prediction error on pattern memory tasks demonstrates that architectural modifications guided by transparency metrics can substantially improve performance. However, the catastrophic degradation on Lorenz data suggests an important principle: when the network fails, it fails transparently. The ability to directly observe pattern formation and usage makes failure modes immediately apparent.

This characteristic of transparent failure has significant implications for AI safety. The capacity to observe pattern formation during training enables direct monitoring of what features a network is learning and how it is using them. This could be particularly valuable as models scale up in size and capability, potentially providing early warning signals of undesired learning dynamics or emergent behaviors.

Several practical extensions could improve the architecture's generality. Language task performance might be enhanced through preprocessing to highlight sequential patterns, or through hybrid architectures combining hierarchical and flat pattern representations. The network's inherent transparency provides clear feedback about the effectiveness of such modifications.

The results suggest that the traditional trade-off between neural network performance and interpretability may not be fundamental. More importantly, they demonstrate that it's possible to design neural networks that are transparent by construction rather than requiring post-hoc interpretation methods. This architectural approach could inform the development of more controllable and verifiable AI systems, where understanding internal representations is as critical as achieving performance benchmarks.

## References

1. Python Software Foundation. Python Language Reference, version 3.10.16. Available at http://www.python.org
2. Anthropic. Claude 3.5 Sonnet (October 2024).

## Code Availability
Implementation code and examples are available at: https://github.com/mac-n/predictiveprocessing_nn
