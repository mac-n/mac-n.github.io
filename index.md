# Training-Transparent Neural Networks with Learned Interpretable Features

Niamh McCombe

January 2025

![my email](/images/myemail.png)

## Abstract

This paper introduces a neural network architecture that maintains interpretability during training while achieving competitive or superior performance on specific sequential prediction tasks. The Pattern Predictive Network (PPN) uses learned pattern dictionaries and prediction-error based routing to create transparent information flow throughout the network. In tests against standard neural architectures, the base PPN achieved a 76% reduction in prediction error on chaotic sequence data while maintaining interpretable internal representations. A hierarchical variant demonstrated a 32% improvement on pattern memory tasks, though with performance trade-offs on other data types. These results suggest that the traditional trade-off between neural network performance and interpretability may not be fundamental, and demonstrate the feasibility of designing neural networks that are transparent by construction rather than requiring post-hoc interpretation methods.

## 1. Introduction

Interpretability is a core concern in AI research. Neural networks, the basic structure underlying almost all recent AI breakthroughs, are typically considered "black boxes" in which it is difficult or impossible to understand the underlying decision-making process. As models grow larger and more capable, the complexity of understanding their decision making increases. The implications range from large-scale concerns about the safety and possible underlying intentions of the most capable AIs to concerns about life-altering administrative decisions being made in areas such as law, healthcare, and insurance without any facility for affected individuals to query the underlying reasoning.

While several approaches to neural network interpretability exist, most rely on post-hoc interpretation tools like SHAP or LIME or dictionary learning, which attempt to explain patterns of decisions made by already-trained networks.

This work presents a novel neural network architecture that achieves transparency as an emergent property of the training process rather than through post-hoc interpretation methods. The Pattern Predictive Network (PPN) uses two key innovations working in concert: pattern dictionaries that learn interpretable features during training, and prediction-based routing that makes information flow observable throughout the network. The architecture demonstrates that transparency need not come at the cost of performance, achieving substantial improvements over standard neural networks on specific tasks while maintaining interpretable internal representations.

Two key features distinguish the PPN architecture:

1. **Pattern Dictionary**
   - Learns interpretable patterns at each layer
   - Enables direct visualization of learned features
   - Tracks and exposes pattern usage statistics

2. **Prediction Error Based Routing**
   - Each layer predicts the next layer's compression of its information
   - If prediction is accurate, information routes to the pre-output layer rather than the next layer
   - Routing decisions are modulated by uncertainty
   - Key intuition: if information doesn't surprise the next layer, why tell it?

While other approaches have used predictive coding principles to create efficient training algorithms with local computations, the PPN architecture uniquely applies these principles to achieve interpretability during training. The pattern dictionaries provide explicit visualization of learned features, demonstrated clearly through their specialization to different regions of the Lorenz attractor's state space, while the prediction-based routing reveals how these features are used to process information. These mechanisms work together to create a network whose internal representations and routing decisions are directly observable throughout training and inference. This represents a shift from trying to interpret black-box models after the fact, to building models that are inherently interpretable without sacrificing performance.

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

#### 3.2.1 Information Flow and Confidence

The PPN's routing mechanism uses prediction confidence to determine information flow through the network. Each layer attempts to predict how the next layer will compress its output. This prediction generates a confidence value between 0 and 1, which determines whether information continues to the next layer or routes directly to the pre-output layer.

The confidence value represents the network's certainty in its prediction of the next layer's activity. A confidence of 1 would indicate perfect prediction of the next layer's response, while 0 would indicate complete failure to predict. In practice, confidence values tend to converge around 0.5 during stable training, reflecting a balanced state where the network maintains some uncertainty about its predictions while still capturing meaningful patterns.

This convergence to intermediate confidence values is a desirable property - it prevents the network from becoming either overly conservative (routing everything to output) or overly uncertain (passing everything to the next layer). The routing decisions can be observed in Figures 3 and 4, where the relationship between confidence (blue line) and information flow (red and green lines) demonstrates the dynamic balance between direct output contribution and continued processing.

The magnitude of information flow through different routing paths (shown in red for "Continue Up" and green for "To Penultimate") typically shows larger variations than the confidence values, as these represent the actual amount of information being routed rather than prediction certainty. This relationship between steady confidence and varying flow magnitude indicates that the network maintains consistent prediction capability while flexibly routing different amounts of information based on input complexity.

![Figure 3](/images/Fig3.png)
[Figure 3: Pattern usage evolution and information flow in Layer 0. Upper panel shows pattern activation over time; lower panel shows routing decisions between direct output contribution and continued processing.]

![Figure 4](/images/Fig4.png)
[Figure 4: Pattern usage evolution and information flow in Layer 1, demonstrating different pattern specialization and routing dynamics compared to Layer 0.]

Pattern correlation analysis (Figure 5) revealed largely independent pattern development, particularly in lower layers. The pattern usage distribution (Figures 6 and 7) showed clear clustering of activations, suggesting specialization of patterns for distinct sequence features.

![Figure 5](/images/Fig5.png)
[Figure 5: Pattern correlations across layers, showing independence of learned patterns. Red indicates positive correlation, blue indicates negative correlation.]

### 3.3 Hierarchical Extension Results

The hierarchical variant of the PPN architecture showed a dramatically different performance profile. For pattern memory sequences, it achieved significantly better performance (mean loss 1.227, SD = 0.072) compared to the standard network (mean loss 1.809, SD = 0.140; t = 7.380, p < 0.0001), representing a 32% reduction in prediction error.

However, this architectural modification severely impacted performance on the Lorenz data (mean loss 5.725, SD = 1.118) compared to both the standard network (mean loss 0.021, SD = 0.004; t = -10.207, p < 0.00001) and the base PPN. Language modeling showed no significant improvement (mean loss 3.661, SD = 0.649 vs 3.240, SD = 0.339; t = -1.150, p = 0.283).

For language data specifically, pattern activations at word boundaries (Figure 6) and character-pattern associations (Figure 7) revealed limited structural organization, suggesting the current architecture may be suboptimal for discrete symbolic sequences.

![Figure 6](/images/Fig8.png)
[Figure 6: Layer 0 pattern activation at word boundaries, showing limited differentiation between boundary and non-boundary positions.]

![Figure 7](/images/Fig9.png)
[Figure 7: Layer 0 pattern-character associations, demonstrating weak specialization for specific character types.]

## 4. Discussion

This work demonstrates that transparent neural networks can emerge from careful architectural design rather than requiring post-hoc interpretation methods. The Pattern Predictive Network achieves transparency through two mechanisms working in concert: pattern dictionaries that learn interpretable features during training, and prediction-based routing that makes information flow observable. The network's success on the Lorenz system is particularly revealing - the pattern dictionaries develop specialized detectors for different regions of the attractor's state space, providing direct visual evidence that the network learns meaningful structure in the underlying dynamics.

This represents a significant advance: rather than requiring post-hoc analysis, the network's internal representations and decision-making processes are observable by design throughout training and inference.

While the base architecture showed substantial improvements on chaotic sequence prediction, the hierarchical variant demonstrated both the potential and limitations of this approach: markedly improved performance on pattern memory tasks but degraded performance on Lorenz data. Crucially, these failure modes were  apparent through the network's transparent architecture. This characteristic of transparent failure - where the network's struggles can be directly observed and understood - represents a significant advance over traditional architectures where failure modes often remain opaque.

Several promising directions emerge from these results. The performance differences between architectural variants suggest that hybrid approaches might be possible, combining the continuous pattern recognition capabilities of the base architecture with the hierarchical features that proved effective for discrete sequences. For language processing specifically, preprocessing approaches that highlight underlying sequential patterns might help bridge current performance gaps. The architecture's transparency makes such modifications particularly tractable, as their effects on pattern formation and usage can be directly observed.

The potential scaling properties of this architecture present another important area for investigation. As network size increases, the pattern dictionaries and routing mechanisms should theoretically maintain their transparency, enabling selective pruning and optimization based on observed pattern usage. This could lead to more efficient training and deployment of large-scale models where computational efficiency is crucial.

More broadly, this work suggests a fundamental shift in neural network design: from trying to peer into black boxes after the fact, to building systems that are transparent by construction. As AI models grow in complexity, the ability to directly observe pattern formation and usage during training could provide crucial insights into learning dynamics and emergent behaviors. Such inherently transparent architectures could even enable novel forms of model interaction and composition, as their internal representations would be accessible and interpretable to each other during operation.



## References

1. Python Software Foundation. Python Language Reference, version 3.10.16. Available at http://www.python.org
2. Anthropic. Claude 3.5 Sonnet (October 2024).
3. https://www.ni-hpc.ac.uk/CaseStudies/PredictiveCodingfortrainingdeepneuralnetworks/
4.https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
5. (references section tbd)

## Code Availability
Implementation code and examples are available at: https://github.com/mac-n/predictiveprocessing_nn
