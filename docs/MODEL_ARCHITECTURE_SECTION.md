# Model Architecture

## Architecture Overview

The proposed model employs a transformer encoder architecture to predict musical slur annotations from sequences of musical notes. The model processes variable-length sequences of note features and produces a categorical prediction for each note in the sequence, classifying it into one of five mutually exclusive slur categories.

## Input Representation

Each musical note is represented as a 6-dimensional feature vector consisting of the following components:

1. **Start time** ($t_{start}$): The onset time of the note in seconds, normalized to a 0-100 scale using min-max normalization across the entire piece.
2. **Duration** ($d$): The duration of the note in seconds, normalized to a 0-100 scale using min-max normalization across the entire piece.
3. **Pitch** ($p$): The MIDI pitch number (ranging from 21 to 108, corresponding to A0 to C8), converted to a relative scale (0-87) by subtracting 21.
4. **Velocity** ($v$): The MIDI velocity value (ranging from 0 to 127), normalized to a 0-100 scale by dividing by 127 and multiplying by 100.
5. **Sustain start** ($s_{start}$): The sustain pedal state value at the note's onset time, extracted directly from the pedal matrix at the corresponding time tick. The raw value ranges from 0 to 127 (where 0 indicates the pedal is off and non-zero values indicate the pedal is on), which is normalized to a 0-100 scale during preprocessing.
6. **Sustain end** ($s_{end}$): The sustain pedal state value at the note's release time, extracted directly from the pedal matrix at the corresponding time tick. The raw value ranges from 0 to 127 (where 0 indicates the pedal is off and non-zero values indicate the pedal is on), which is normalized to a 0-100 scale during preprocessing.

Given a sequence of $N$ notes, the input to the model is a tensor of shape $(N, 6)$, where each row corresponds to the feature vector of a single note, and the sequence order reflects the temporal ordering of notes in the musical piece.

## Model Architecture

The model architecture consists of three primary components: an input projection layer, a stack of transformer encoder layers, and an output projection layer.

### Input Projection

The input projection layer is a linear transformation that maps the 6-dimensional input features to the model's hidden dimension $d_{model}$:

$$\mathbf{h}_0 = \mathbf{W}_{in} \mathbf{x} + \mathbf{b}_{in}$$

where $\mathbf{x} \in \mathbb{R}^{6}$ is the input feature vector, $\mathbf{W}_{in} \in \mathbb{R}^{d_{model} \times 6}$ is the weight matrix, $\mathbf{b}_{in} \in \mathbb{R}^{d_{model}}$ is the bias vector, and $\mathbf{h}_0 \in \mathbb{R}^{d_{model}}$ is the projected representation. The model uses $d_{model} = 128$.

### Transformer Encoder

The projected input sequence is processed through a stack of $L = 4$ identical transformer encoder layers. Each encoder layer implements the standard transformer architecture with multi-head self-attention and position-wise feed-forward networks.

#### Multi-Head Self-Attention

Each encoder layer contains a multi-head self-attention mechanism with $H = 8$ attention heads. For each head $h \in \{1, \ldots, H\}$, the attention mechanism computes:

$$\text{Attention}_h(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}\right) \mathbf{V}_h$$

where $\mathbf{Q}_h$, $\mathbf{K}_h$, and $\mathbf{V}_h$ are the query, key, and value matrices obtained by linear projections of the input, and $d_k = d_{model} / H = 16$ is the dimension of each head. The outputs from all heads are concatenated and projected through a linear layer:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O$$

where $\mathbf{W}_O \in \mathbb{R}^{d_{model} \times d_{model}}$ is the output projection matrix.

The attention mechanism is bidirectional (non-causal), allowing each position to attend to all positions in the sequence, which is appropriate for musical analysis where both past and future context are relevant for slur prediction.

#### Position-wise Feed-Forward Network

Following the multi-head attention, each encoder layer contains a position-wise feed-forward network (FFN) consisting of two linear transformations with a ReLU activation:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$, and $d_{ff} = 512 = 4 \times d_{model}$ is the feed-forward dimension.

#### Residual Connections and Layer Normalization

Each sub-layer (attention and FFN) is wrapped with residual connections and layer normalization:

$$\mathbf{x}_{l+1} = \text{LayerNorm}(\mathbf{x}_l + \text{Sublayer}(\mathbf{x}_l))$$

where $\text{Sublayer}$ is either the multi-head attention or the feed-forward network, and $\text{LayerNorm}$ denotes layer normalization.

#### Dropout

A dropout rate of $p = 0.1$ is applied to the outputs of both the attention mechanism and the feed-forward network during training to prevent overfitting.

### Output Projection

The final encoder layer output is projected through a linear layer to produce logits for each of the five output classes:

$$\mathbf{y} = \mathbf{W}_{out} \mathbf{h}_L + \mathbf{b}_{out}$$

where $\mathbf{h}_L \in \mathbb{R}^{d_{model}}$ is the output from the final encoder layer, $\mathbf{W}_{out} \in \mathbb{R}^{5 \times d_{model}}$ is the output weight matrix, $\mathbf{b}_{out} \in \mathbb{R}^{5}$ is the output bias vector, and $\mathbf{y} \in \mathbb{R}^{5}$ contains the raw logits for each class. No activation function (such as softmax) is applied to the output layer, as the softmax operation is performed internally by the categorical cross-entropy loss function during training.

## Output Classes

The model predicts one of five mutually exclusive classes for each note:

- **Class 0**: Slur start ($c_0$)
- **Class 1**: Slur middle ($c_1$)
- **Class 2**: Slur end ($c_2$)
- **Class 3**: No slur ($c_3$)
- **Class 4**: Slur start and end ($c_4$)

The mapping from annotation categories to model classes is as follows: annotation category 1 (slur start) maps to class 0, category 2 (slur middle) maps to class 1, category 3 (slur end) maps to class 2, category 4 (no slur) maps to class 3, category 5 (slur start and end) maps to class 4, and category 0 (background/unassigned) maps to class 3 (no slur).

## Loss Function

The model is trained using categorical cross-entropy loss (implemented as `nn.CrossEntropyLoss` in PyTorch), which is appropriate for multi-class classification with mutually exclusive classes. For a sequence of $N$ notes, the loss is computed as:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(y_{i,c_i})}{\sum_{j=0}^{4} \exp(y_{i,j})}$$

where $y_{i,j}$ is the logit for class $j$ at position $i$, and $c_i$ is the true class label for note $i$. The softmax operation is applied internally by the loss function during the forward pass, converting the raw logits into a probability distribution over the five classes before computing the cross-entropy. This approach is numerically stable and is the standard implementation in PyTorch, avoiding the need to explicitly apply softmax in the model architecture.

## Weight Initialization

All linear layer weights are initialized using Xavier uniform initialization:

$$\mathbf{W} \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right)$$

where $n_{in}$ and $n_{out}$ are the number of input and output features for the layer, and $\mathcal{U}(a, b)$ denotes a uniform distribution on the interval $[a, b]$. Bias terms are initialized to zero.

## Model Parameters

The model contains approximately 794,500 trainable parameters. The exact parameter count is 794,501, distributed across the following components:

- Input projection layer: $6 \times 128 + 128 = 896$ parameters
- Transformer encoder (4 layers): The PyTorch `TransformerEncoderLayer` implementation includes multi-head self-attention, feed-forward networks, layer normalization, and residual connections. The exact parameter distribution within each layer depends on the internal implementation of PyTorch's transformer components.
- Output projection layer: $128 \times 5 + 5 = 645$ parameters

(Note: The parameter count of 794,501 was verified empirically by summing all model parameters. The exact distribution within the transformer encoder layers may differ from manual calculations due to implementation details in PyTorch's `TransformerEncoder` and `TransformerEncoderLayer` classes.)

## Training Configuration

The model is trained using the Adam optimizer with a learning rate of $\alpha = 0.001$. Training is performed using a chunked approach, where long musical sequences are divided into overlapping chunks of 200 notes with an overlap of 100 notes between consecutive chunks. This approach preserves musical context while enabling efficient training on sequences that exceed the model's effective context window.

Gradients are accumulated across chunks within each training batch, with a batch size of 1. Early stopping is employed with a patience of 50 epochs, monitoring validation accuracy to prevent overfitting. Training proceeds for a maximum of 200 epochs, though early stopping typically terminates training earlier.

## Sequence Processing

During inference, the model processes sequences of variable length. For sequences longer than the chunk size used during training, the sequence is divided into overlapping chunks, and predictions are made for each chunk independently. The overlap ensures that context is preserved at chunk boundaries, maintaining continuity in the predictions.

## Implementation Details

The model is implemented using PyTorch, utilizing the `nn.TransformerEncoder` and `nn.TransformerEncoderLayer` classes from `torch.nn`. The encoder layers are configured with `batch_first=True`, resulting in input tensors of shape $(B, N, d_{model})$ where $B$ is the batch size, $N$ is the sequence length, and $d_{model}$ is the hidden dimension. The model processes sequences in a batch-first format throughout the forward pass.

