##### A B S T R A C T

```
License plate recognition in open environments is widely applicable across various domains;
however, the diversity of license plate types and imaging conditions presents significant chal-
lenges. To address the limitations encountered by CNN and CRNN-based approaches in license
plate recognition, this paper proposes a unified solution that integrates a lightweight visual
encoder with a text decoder, within a pre-training framework tailored for single and double-
line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we
constructed a single/double-line license plate dataset by synthesizing images, applying texture
mapping onto real scenes, and blending them with authentic license plate images. Furthermore,
to enhance the system’s recognition accuracy, we introduce a perspective correction network
(PTN) that employs license plate corner coordinate regression as an implicit variable, supervised
by license plate view classification information. This network offers improved stability, inter-
pretability, and low annotation costs. The proposed algorithm achieves an average recognition
accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When
evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the
double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with
processing speeds reaching up to 167 frames per second, indicating strong practical applicability.
```
### 1. Introduction

```
Open license plate recognition (LPR) technology boasts a broad range of applications, including parking manage-
ment, road traffic monitoring, toll station automation, and forensic evidence collection. Its primary advantage lies in the
ability to operate without imposing additional constraints or restrictions on vehicles, making it applicable even when
vehicles are in motion and captured at significant angles. However, in real-world open environments, LPR continues
to face numerous challenges that demand resolution [1][2][3][4]. These challenges stem from complex environmental
lighting, variable weather conditions, and issues such as license plate soiling, particularly when the capture angle is
uncontrolled.
Convolutional Neural Networks (CNNs) [5] and Convolutional Recurrent Neural Networks (CRNNs) [6] are widely
adopted for extracting character features in license plate recognition. Via convolutional operations, these networks
process characters in license plate images sequentially. As convolutional kernels are translated, a single character
yields multiple feature outputs, each derived from image regions covered by different receptive fields. However,
the tight arrangement of characters often leads to feature entanglement, resulting in the interweaving of character
features within these regions. To address this issue, CRNN-based license plate recognition algorithms commonly
incorporate Long Short-Term Memory (LSTM) or Bidirectional LSTM [7]networks. These architectures infer the
most probable character class based on multiple local features extracted from each character, thereby improving
recognition accuracy. Alternatively, pure CNN-based networks, such as LPRNet [8], employ global lateral convolution
operations, enabling the model to capture more comprehensive character features and thus enhance understanding
⋆ Supported by the Open Fund of Hubei Key Laboratory of Intelligent Visual Monitoring for Hydropower Engineering (China Three Gorges
University, 2022SDSJ03).
∗Principal corresponding author
∗∗Corresponding author
xgz@ctgu.edu.cn(G. Xu);kz@ctgu.edu.cn(Z. Ke);zpc@ctgu.edu.cn(P. Zuo);Bangjun.Lei@ieee.org(B. Lei)
ORCID(s):0009-0004-7154-9831(G. Xu)
```
```
: Preprint submitted to Elsevier Page 1 of 28
```
# arXiv:2507.17335v1 [cs.CV] 23 Jul 2025


of inter-character relationships. Furthermore, these models often integrate the Connectionist Temporal Classification
(CTC) loss function and dynamic decoding strategies [9], constructing a probability distribution space for character
sequences. This approach resolves the automatic alignment of variable-length character sequences and further enhances
recognition precision.
However, when license plate characters exhibit distortions due to camera angle, leading to variations in character
size or glyph deformation, the aforementioned algorithms are prone to character omission or insertion errors, such as
misidentifying a seven-character plate as an eight-character plate, or vice versa. This issue is particularly prevalent
in license plate recognition schemes employing rectangular bounding boxes for localization [10]. In contrast, license
plate vertex localization techniques, which obtain the corner coordinates of the plate, enable more accurate license plate
rectification [11]. Nevertheless, this approach places a higher demand on training data annotation and the effectiveness
of plate rectification is highly dependent on the precision of vertex localization.
To achieve adaptive license plate correction tailored for recognition tasks, an increasing number of license plate
recognition schemes incorporate the integration of correction and recognition modules. Reference [8] combines Spatial
Transform Networks (STN) [12] with recognition networks to realize adaptive correction of license plate images
through learned transformations. However, since STN relies on supervisory feedback from subsequent recognition
networks to estimate affine transformation parameters, it requires pretraining of the recognition network, and only
after reaching a certain performance level can it be integrated with the STN. Furthermore, due to the interdependent
nature of perspective transformation parameters, employing STN for perspective correction often leads to issues such
as divergence or non-convergence during training. Consequently, this approach proves difficult to adapt for license
plate recognition in open environments with varying shooting angles.
On the other hand, current research on Chinese license plate recognition primarily focuses on single-line license
plates, with limited attention given to double-line Chinese license plates [13][14]. However, double-line plates are
commonly used on vehicles such as trucks, buses, and trailers. This is partly due to the limitations of Convolutional
Recurrent Neural Network (CRNN) architectures in processing vertical spatial sequence information, making it
difficult to effectively model spatial dependencies in the vertical direction. Traditional Connectionist Temporal
Classification (CTC) decoding relies on one-dimensional sequence features and cannot effectively encode the spatial
relationships of characters in double-line plates. Furthermore, the lack of diverse license plate datasets, especially
those containing double-line plate images, severely restricts the research and development of Chinese double-line
license plate recognition.
With continual advancements in vision Transformer algorithms, the OCR solution based on the Transformer
encoder-decoder architecture, TrOCR [15], offers a novel approach to optical character recognition by leveraging its
unique global self-attention mechanism. Compared to the limitations of traditional CNN/CRNN+CTC architectures
in modeling one-dimensional sequences, the self-attention mechanism of Transformers can more effectively capture
the relationships between arbitrary image regions within optical character images, thereby enhancing the accuracy of
character recognition. Furthermore, Transformer text models pre-trained on large-scale unannotated datasets encode
semantic priors, which can somewhat mitigate issues related to insufficient training data for double-line license plates.
However, TrOCR suffers from disadvantages such as a large model size and low computational efficiency.
Building upon the aforementioned considerations and inspired by Spatial Transformer Networks (STN) and
TrOCR, this paper introduces a novel license plate recognition framework that leverages a Transformer-based encoder-
decoder architecture. We propose TransLPRNet, a lightweight model that integrates joint visual and textual pre-training
for both single-line and double-line license plate recognition. Additionally, we design a License Plate Perspective
Transformation Network (PTN), driven by view classification information that distinguishes whether an image depicts
a frontal view of a license plate. An extended dataset is constructed by combining synthetically generated double-line
license plate images with the CCPD dataset; this augmentation employs an information redundancy removal strategy
to enrich the dataset with double-line plate images while preserving the completeness and scale of the original CCPD
training set. Compared to other mainstream license plate recognition algorithms, our approach achieves superior results.
The main contributions of this work are summarized as follows:

1. A lightweight vision-language hybrid license plate recognition network, TransLPRNet, is proposed. It employs a
pre-trained MobileViTv3 [16] model as the encoder and a MiniMLv2 [17] model as the decoder. Experimental results
show that, compared to TrOCR, TransLPRNet significantly reduces the number of model parameters and computational
complexity, while also achieving notable improvements in recognition accuracy and inference speed. Additionally, this
network demonstrates excellent performance and strong adaptability in recognizing single-line and double-line Chinese
license plates.


2. A Perspective Transformation Network (PTN) for automatic rectification is introduced, which leverages weak
supervision from a lightweight binary classification network to enable rapid automatic correction of various license
plate types. PTN effectively rectifies perspective distortions in license plates captured from different scenes by
identifying frontal-view images, thereby significantly reducing annotation costs.
3. To address the lack of double-line license plate datasets in unconstrained environments, this study employs
a texture-mapping approach, overlaying generated double-line license plates onto real scenes and mixing them with
authentic license plate images. This method constructs a diverse double/single-line license plate dataset. Furthermore,
by compressing redundant information in the original CCPD dataset without increasing the number of training samples
and while preserving all original data, the study extends the CCPD dataset to include images of double-line license
plates.
The remainder of this paper is organized as follows. Section 2 summarizes the advantages and disadvantages
of existing license plate recognition and correction algorithms, and introduces relevant publicly available datasets.
Section 3 presents the proposed algorithms and the dataset constructed in this study. Section 4 reports the experimental
results and provides a detailed analysis. Finally, the conclusion discusses the key findings and outlines future research
directions.

### 2. Related work

#### 2.1. License Plate Recognition

For license plate recognition (LPR) tasks in open environments, current mainstream algorithms often employ OCR
networks based on CNNs and CRNNs (CNN+RNN). Zherzdev et al. [8] pioneered LPRNet, a real-time LPR model that
dispenses with the RNN component. This model utilizes a lightweight CNN backbone and replaces traditional LSTMs
with wide convolutions for capturing local contextual information of characters. Coupled with the Connectionist
Temporal Classification (CTC) loss function [9], LPRNet directly outputs variable-length character sequences. Their
method achieved an average recognition accuracy of 95% on a Chinese LPR dataset, demonstrating a favorable balance
between recognition speed and accuracy. Xu et al. [18]constructed and open-sourced CCPD, the first comprehensive
LPR dataset covering complex scenarios. They also proposed RPNet, a unified network architecture that integrates
license plate detection and character recognition into an end-to-end pipeline. In the recognition stage, RPNet employs
CRNN (CNN + BiLSTM + CTC) to directly recognize character sequences from Regions of Interest (ROIs) regressed
from the detection box.
However, these methods often encounter significant challenges when processing double-line license plates.
Traditional Convolutional Neural Network (CNN) or Convolutional Recurrent Neural Network (CRNN) architectures
typically struggle to effectively handle multi-line information, particularly when dealing with uneven character
distributions and variable numbers of lines. This limitation impedes the network’s ability to capture contextual
dependencies, consequently hindering the accuracy and robustness of double-line license plate recognition in complex
scenarios. To address this limitation, Qin et al. [19] proposed a unified framework for recognizing both single-line
and double-line license plates. Their approach leverages an improved lightweight CNN for efficient feature extraction
and employs a multi-task learning strategy to simultaneously perform license plate classification and character
recognition. Finally, the recognition task is formulated as a sequence labeling problem and solved using Connectionist
Temporal Classification (CTC) loss. While effective for some cases, these methods often require segmenting the
double-line license plate into upper and lower regions and subsequently integrating the individual recognition results.
Consequently, their performance degrades significantly when processing highly skewed or tilted license plates, often
leading to the omission of some characters during recognition.
Compared to traditional OCR methods, deep learning-based OCR models like PaddleOCR [20] have significantly
improved recognition accuracy and computational efficiency through optimized algorithms and the incorporation of
novel techniques, particularly excelling in the processing of double-line license plates. Li et al. [21]proposed PP-
OCRv3, which, building upon its predecessor, introduces several enhancements, including the lightweight SVTR-
LCNet recognition network, an attention-guided CTC training strategy, and diverse data augmentation methods,
enabling a dynamic balance between accuracy and speed.
Although the aforementioned lightweight solutions effectively mitigate the deployment efficiency bottlenecks
of traditional models, their backbone architectures still rely on CNNs, which limits their capacity to model long-
range dependencies and hampers the comprehensive representation of complex semantic and contextual relationships
between characters. Additionally, such approaches face challenges in recognizing double-line license plates. To further


enhance the expressive power and robustness of recognition models, researchers have recently begun exploring unified
vision-text modeling frameworks based on Transformer architectures. These methods employ collaborative modeling
between visual encoders and text decoders, incorporating large-scale pre-training and fine-tuning mechanisms to
achieve stronger sequence modeling and cross-modal feature alignment. For example, the end-to-end text recognition
model TrOCR proposed in [15] utilizes collaborative image and text Transformers to generate complex text sequences,
thereby improving the modeling capability for intricate character sequences. However, Transformer-based recognition
methods generally depend on large pre-trained models, which demand substantial computational resources and have
slower inference speeds, presenting significant challenges for deployment in real-time recognition applications.
In recent years, with the development of lightweight vision-language Transformer models, networks such as
MobileViT [22] and MiniLM [23] have demonstrated promising performance and deployment efficiency across various
vision and text modeling tasks. Specifically, MobileViT achieves efficient encoding of image structural information
by integrating the local perception of CNNs with the global modeling capability of Transformers. MiniLM, as a
compact text Transformer, possesses robust character sequence modeling capabilities with a small model size and fast
inference speed. Building on this, we propose a lightweight end-to-end license plate recognition network architecture
that jointly utilizes MobileViTv3 [16] and MiniLMv2 [17], aiming to simultaneously balance recognition accuracy and
deployment efficiency, making it suitable for real-time license plate recognition tasks in resource-constrained scenarios.

#### 2.2. License Plate Spatial Transformation Correction

Current OCR architectures have achieved significant progress in license plate recognition tasks; however, license
plate images captured in real-world, open environments often exhibit issues such as angular tilt and geometric
distortions, which pose challenges to subsequent character recognition in terms of accuracy and robustness. To improve
recognition performance, license plate image rectification has increasingly become a critical preprocessing step and
an active area of research.
Currently, the mainstream approaches for license plate rectification can be broadly categorized into two types. The
first involves obtaining the four corner coordinates of the license plate region through a plate localization module,
and then computing a perspective transformation matrix based on these coordinates and the desired rectified image
size to achieve rectification. For example, Kundrotas and colleagues [24] proposed a lightweight network to detect the
four corners of the license plate and employed a perspective inverse transformation to perform geometric correction,
thereby simplifying subsequent character recognition tasks. Their method utilizes an improved Hourglass network as a
feature extractor and achieved an average recognition accuracy of 96.19% on a Chinese license plate dataset. However,
such approaches rely heavily on high-quality corner annotations and treat the rectification and recognition modules
as separate entities. This separation makes it challenging to optimize the entire process end-to-end for stable spatial
correction aimed at license plate character recognition.
The second category encompasses methods leveraging learnable Spatial Transformer Networks (STNs) [12]. These
approaches employ a machine learning paradigm, adjusting network parameters by backpropagating the error signal
derived from the subsequent license plate recognition network. In contrast to the first category, these methods obviate
the need for manual annotation of corner points, thereby exhibiting enhanced adaptability. Xiao et al. [25] utilized
the YOLOv2 detector for license plate detection and proposed the ICSTN-CRNN model. This model integrates
a Thin-Plate Spline-based Spatial Transformer Network (STN) to achieve automatic license plate rectification and
recognition, demonstrating robust performance across multiple datasets. Furthermore, Akshay Bakshi et al. [26]
adopted a hybrid approach, combining Spatial Transformer Networks (STNs) with Convolutional Neural Networks
(CNNs). They proposed an automatic license plate recognition system capable of rectification and character recognition
under complex environmental conditions and multi-angle captures, achieving high recognition accuracy across datasets
encompassing diverse regions and varying acquisition conditions.
Although the aforementioned methods partially alleviate issues related to angular deviation and deformation,
the Spatial Transformer Networks (STNs) employed in these algorithms are primarily limited to spatial correction
involving affine transformations. However, in open-world environments, license plate images often exhibit perspective
distortions, rendering affine-based STNs insufficient for effective rectification. While the solution proposed in reference
[25] can address perspective transformation challenges by regressing 110 points, the large number of regression points
may lead to model overfitting. This, in turn, can cause the model to exhibit instability and prediction fluctuations during
the testing phase.


#### 2.3. License Plate Datasets

The license plate dataset is a crucial component of license plate recognition tasks. A high-quality, diverse Chinese
license plate dataset provides a solid foundation for model training and evaluation. Currently, the main Chinese license
plate datasets include CCPD [18], CLPD [27] and LSV-LP [28]. Among these, the CCPD dataset is the most widely
used Chinese license plate recognition dataset, comprising approximately 300,000 labeled single-line license plate
images, which are divided into multiple subsets based on environmental variations such as Rotation, Tilt, Blur, and
Weather conditions. The CLPD dataset, released by the Institute of Automation, Chinese Academy of Sciences,
contains 1,200 license plate images representing various provinces across China. Due to its relatively small size and
limited scenarios, its application scope is more constrained. LSV-LP is a Chinese license plate recognition dataset
containing approximately 400,000 images, covering various license plate types and common noise factors, and is
frequently used to evaluate the performance of recognition algorithms in complex scenes. Compared to CCPD, LSV-LP
still exhibits certain deficiencies in annotation accuracy, image clarity, and coverage of specialized scenarios.
Although current Chinese license plate datasets offer a foundation for training and evaluation of recognition
algorithms, certain shortcomings necessitate further attention. A significant limitation stems from label errors within
some datasets. Because these datasets are annotated manually, a degree of mislabeling is unavoidable. Another critical
deficiency is the insufficient proportion of double-line license plates. Given the widespread use of double-line plates
in practice, this imbalance makes it challenging to effectively evaluate the performance of license plate recognition
algorithms specifically on this important class of license plates.

### 3. Our method

To address the challenges of license plate recognition in open environments, this paper first introduces a novel dual-
style license plate recognition network, TransLPRNet. The model leverages a lightweight visual encoder, MobileViTv
[16], combined with a pre-trained text decoder, MiniMLv2 [17], to enhance feature extraction and recognition
accuracy. To further improve performance, a versatile perspective transformation space auto-correction network, PTN,
is designed to correct images captured from various angles and deformations. Additionally, considering the lack of non-
constrained environment datasets for dual-style license plates, we propose a dataset construction method. This involves
synthesizing dual-style plates using templates and blur transformations, then seamlessly replacing the original license
plates in redundant images within the CCPD dataset. These redundant images are highly similar and contain limited
unique information, ensuring minimal impact on the overall dataset diversity. This approach enables the construction
of suitable datasets for non-constrained environment recognition. Refer to Figure 1 for the system architecture.

```
CCPD Extend
```
```
Perspective
Transform&blurring
```
```
Synth.Double-line LPs
```
```
LP location
Disturbance PTN:Perspective Transformation Network
```
```
Front view LPs
classifier*
```
```
Weak
Supervision
Signals
```
```
*Only used for PTN training
```
```
Visual Encoder
```
```
Text Decoder
```
```
皖AVL987 皖AZJ
```
```
沪MAT98U 藏CZ952T
```
```
赣27BE1B挂 津334RN
```
```
TransLPRNet
```
```
Recognition results
```
```
Hybrid LP Dataset
```
```
Figure 1:System solution diagram
```

```
Input Image
```
```
Linear +Softmax
```
```
Multi-Head
AttentionCross
```
```
Add & Norm
```
```
Masked
MultiAttention-Head
```
```
Add & Norm
```
```
FFN
```
```
Add & Norm
```
```
皖 J M H 7 8 K EOS
```
```
Conv- 3 × 3
```
```
Inversted Residual
Block ×
```
```
Decoder output
```
```
Positional Encoding
```
```
MiniLMv2 Block
```
```
×
```
```
Local RepresentationUnit
```
```
Inversted ResidualBlock
```
```
Conv- 1 × 1
```
```
MobileViTBlock
```
```
×
```
```
×
```
```
Inversted ResidualBlock
```
```
MobileViTBlock ×
```
```
Patch Embedding
```
```
MobileViTBlock
```
```
Linear
```
```
Input Image
```
```
Linear +Softmax
```
```
Multi-Head
AttentionCross
```
```
Add & Norm
```
```
Masked
MultiAttention-Head
```
```
Add & Norm
```
```
FFN
```
```
Add & Norm
```
```
皖 J M H 7 8 K EOS
```
```
Conv- 3 × 3
```
```
Inversted Residual
Block ×
```
```
Decoder output
```
```
Positional Encoding
```
```
MiniLMv2 Block
```
```
×
```
```
Local RepresentationUnit
```
```
Inversted ResidualBlock
```
```
Conv- 1 × 1
```
```
MobileViTBlock
```
```
×
```
```
×
```
```
Inversted ResidualBlock
```
```
MobileViTBlock ×
```
```
Patch Embedding
```
```
MobileViTBlock
```
```
Linear
```
```
Output
Embedding
Outputs
(shifted right)
```
```
Global Representation
Unit
```
```
Fusion Unit
```
```
Figure 2:TransLPRNet network structure diagram
```
#### 3.1. TransLPRNet

The network architecture of TransLPRNet, as shown in Figure 2, primarily consists of two core modules: a visual
encoder based on MobileViTv3 and a text decoder based on MiniMLv2. The visual encoder employs an alternating
hybrid approach that integrates convolutional modules and Transformer modules, effectively enabling efficient intra-
window and inter-window information exchange. Subsequently, the output of the visual encoder is processed through a
linear transformation to achieve compatibility with the decoder. In the decoder, a cross-attention mechanism is utilized,
functioning through multiple iterative steps to progressively decode the encoded tokens from the license plate images,
thereby achieving high-precision license plate recognition. The overall architecture leverages the strengths of both
modules, ensuring both the accuracy and efficiency of the system.

**_3.1.1. Lightweight Encoder based on MobileViTv_**
Figure 3 illustrates the structure of the license plate visual encoder module used in this study, based on the
MobileViTv3-small [16] model. The inverted residual blocks(IRB), outlined by gray dashed lines, are adapted from
MobileNet [29]. These blocks enable efficient local feature extraction by first expanding the channel dimensions, then
compressing them, combined with depthwise separable convolution.
Taking the first IRB, marked by the gray dashed outline, as an example: it begins with a 1×1 convolution that
expands the number of channels from 16 to a higher dimension (specifically, 16 multiplied by the expansion ratio),
which enhances the feature representation. In MobileViTv3, the expansion ratio for all IRB is 4. Next, a 3×3 depthwise
convolution layer extracts spatial information. Finally, a 1×1 convolution reduces the number of channels to 32, which
is the output. Since the first IRB has different input and output channel numbers, MobileViTv3 omits the residual
connection (shown by the blue dashed arrow inside the gray dashed box) to reduce computation associated with
downsampling.
In this encoder, the first convolutional layer within IRB 2, 5, 6, and 7 has a stride of 2, resulting in downsampling.
The 3rd and 4th IRB are identical cascade modules; their internal first convolution layers have a stride of 1, and since
their input and output channels are the same, residual connections are used in these blocks (as shown by the blue dashed
arrow in module 1 in the figure).
As indicated by the blue dashed box in Figure 3, the MobileViTv3 module is composed of three main parts: a
local representation unit, a global representation unit, and a fusion module. The local representation unit is designed


```
Conv-3×3 IRB
```
```
EmbeddingPatch
```
```
MobileViT block
L=
224×224 h=w=
```
```
MobileViT block
112×
```
```
L=
28×28 h=w=
```
```
Downsampling
Image preprocessing
```
```
Residual Connection
```
```
MobileViT block
```
(^28128)
28
IRB IRB
ConvRelu6-1×1, DWConvRelu6-3×3, ConvLinear-1×1,
Module 1
28
(^28) A 128 C
224×
Linear
IRB IRB
56×56 14×14 h=w=
IRB MobileViT block Conv-1×
L=
(^128256)
16 16
Local
Representation
Unit
DWConv

- 3×
    Conv
    - 1×

```
Unfolding
```
```
Transformer
```
```
×L
Folding
```
```
28
```
(^28128)
4 128
196 196
4 64 28
(^28128)
Global Representation
Unit
Conv
28 - 1×
Conv 28 B^256

- 1×

```
Fusion Unit
```
16 → 32 32 → 64 64 → 64 64 → (^128128) → 256 256 → 320
IRB
64 → 64
56×56 56×
16 → 64 64 → 64 64 → 32
IRB: Inverted Residual Block
112×
Figure 3:TransLPRNet encoder network structure diagram
to capture pixel-level local features while retaining fine spatial details, which it achieves using depthwise separable
convolutions. To mitigate the computational cost of the ensuing Transformer, it further employs 1×1 convolutions for
channel dimension compression.
The Global Representation Unit (GRU) is designed for efficient processing of feature information within local
windows. Taking the first MobileViTv3 block in Figure 3 as an example, it receives a feature map of size 28× 28 ×128.
This feature map is first divided into multiple 2×2 local windows and then unfolded. Each 2×2 local window generates
four tokens, each with a dimension of 128. The entire feature map contains 784 such local windows (28×28 / (2×
2) =196), resulting in a total of 784 tokens (196×4 = 784). These tokens are grouped into 196 sets, with each set
comprising four tokens corresponding to an original local window.
The core of the GRU lies in its four parallel Transformer layers. These layers iteratively perform Self-Attention
computations on the four 128-dimensional tokens within each independent local window. This design enables the model
to achieve pixel-level global correlation modeling within a small scope (i.e., inside each local window), effectively
capturing complex dependencies within local regions.
Upon completion of computations across all local windows, the Transformer-processed tokens are folded back
to the original spatial dimensions of the feature map (28× 28 ×128). Finally, a 1×1 convolutional layer is applied to
increase the channel dimension of the features, thereby enhancing the model’s feature representation capability and
providing richer semantic information for subsequent modules.
The Fusion Unit initially concatenates the outputs of the Local and Global Representation Units and employs a
1 ×1 convolution to achieve feature fusion. Subsequently, the fused features are added to the input of the MobileViTv
module, forming a residual connection. Across different MobileViTv3 modules, interaction between local windows
is facilitated via a 3×3 convolution within the inverted residual structure, effectively enabling the modeling of global
information.
To meet the dual demands of accuracy and speed in license plate recognition, we modified the original MobileViTv
architecture. Specifically, we removed its global average pooling layer and fully connected layers, which are typically
used for image classification, and utilized the remaining components as the backbone for our visual encoder’s feature
extraction. To accelerate training, we leveraged the pre-trained weights of this backbone on the ImageNet-1k dataset.
Considering the token dimension requirements of the subsequent decoder, we processed the 7× 7 ×320 feature map
output by the backbone. First, by treating 2×2 windows as patches, we unfolded the feature map into a sequence
containing sixteen feature tokens, each with a dimension of 256. Subsequently, a linear layer adapted the dimension of
each feature token from 256 to 128, resulting in sixteen 128-dimensional tokens, which are then fed into the decoder
(as illustrated in Figure 3). Detailed parameters for each module within the entire encoder part can be found in Table
1, where "↓" denotes a downsampling operation.


```
Table 1
Encoder parameters
```
```
Layer Type Size Repeat Channels Stride
Transformer Parameter
Head Dim Layer
Image 224 × 224 1 3 1
Conv 3×3,↓ 224 × 224 1 16 2
Inverted Residual 112 × 112 1 32 1
Inverted Residual,↓ 112 × 112 1 64 2
Inverted Residual 56 × 56 2 64 1
Inverted Residual,↓ 56 × 56 1 128 2
MobileViT Block 28 × 28 2 128 1 4 8 4
Inverted Residual,↓ 28 × 28 1 256 2
MobileViT Block 14 × 14 4 256 1 4 8 4
Inverted Residual,↓ 14 × 14 1 320 2
MobileViT Block 7 × 7 3 320 1 4 8 4
Conv 1× 1 7 × 7 1 256 1
Patch Embedding 16 × 320 1 256 1
Linear 16 × 128 1 128 1
```
**_3.1.2. MiNiLMv2-Based Lightweight Decoder_**
To ensure the overall lightweight of the model, this paper employs the lightweight MiniLMv2 [17] as the decoder.
As shown in Figure 4, the core component of the decoder consists of four standard Transformer layers [15]. During
training, the MiniLMv2 decoder first uses Masked Multi-Head Attention layers to process the historical information
of the target sequence. Then, it combines this processed information with the feature sequence output from the visual
encoder through Multi-Head Cross Attention layers, ultimately enabling the modeling of sequential dependencies and
contextual constraints within the true label sequence.

```
MiniLMv2 Block Linear
```
```
MiniLMv2 Block
```
```
×
```
```
Masked MultiHead Attention- Add & Norm Cross MultiAttention-Head Add & Norm FFN Add & Norm
```
```
Positional
Encoding
```
```
Softmax
```
```
V
K
Q
```
```
VK
Q
```
```
皖 J M 7 8 K 皖 J M H^78 K
```
```
Encoder input
128
```
```
16
```
```
SOS H EmbeddingOutput EOS
```
```
Q=K=V=(8,128)
```
```
(8,128)
```
```
K=V=(16,128)
```
```
Q=(8,128)
```
```
(8,128)
```
```
128 → 68
```
```
Figure 4:TransLPRNet decoder network structure diagram
```
Specifically, the visual encoder encodes the license plate image into a fixed-length feature sequence _𝑍_ =
{ _𝑍_ 1 _,𝑍_ 2 _,_ ... _,𝑍_ 16 }, where each _𝑍𝑖_ ∈ℝ^128 represents an image-level semantic token. This sequence serves as a
conditional input to the decoder, guiding the decoding process to attend to relevant visual features. Within the decoder,
_𝑍_ interacts with the decoder’s autoregressive sequence through a multi-head cross-attention layer, with the computation
performed as follows:

```
𝐴𝑡𝑡𝑛𝑐𝑟𝑜𝑠𝑠 = 𝑀𝑢𝑙𝑡𝑖𝐻𝑒𝑎𝑑 ( 𝑄 = 𝐻𝑑𝑒𝑐,𝐾 = 𝑉 = 𝑍 ) (1)
```

Here, _𝐻_ decrepresents the hidden state of the decoder’s current layer, which serves as the query ( _𝑄_ ). This allows
the decoder to focus on relevant information within its own hidden state. Meanwhile, _𝑍_ acts as both the key ( _𝐾_ )
and value ( _𝑉_ ), providing image feature information for the attention calculation. This setup enables the decoder to
integrate relevant content from the image features based on its own state. During training, the embeddings _𝑌_ label=
{ _̂𝑦_ 1 _, ̂𝑦_ 2 _,_ ... _, ̂𝑦_ 8 }of the true labels (i.e., the license plate character sequence) are used as the decoder’s autoregressive
input, where _̂𝑦𝑗_ ∈ℝ^128 is a character token embedding.Subsequently, the model employs a Multi-Head Cross Attention
layer to facilitate the interaction between the target information and visual features, yielding contextual information
that provides sufficient semantic support for subsequent predictions.
At the start of the inference process, the MiniLMv2 decoder exclusively receives the feature sequence _𝑍_ from the
visual encoder. This sequence acts as an "initial cue" for the decoder’s prediction of license plate characters, condensing
key visual information from the license plate image, such as character shapes and color distribution, thereby providing
a foundational reference for subsequent decoding. The dynamic autoregressive input, on the other hand, begins with
the start-of-sequence token [SOS]. At each step, a character token, denoted as _̂𝑦𝑡_ , is generated and incorporated as a
new element into the historical generated sequence _𝑌_ gen( _𝑡_ ) = {[SOS] _, ̂𝑦_ 1 _, ̂𝑦_ 2 _,_ ... _, ̂𝑦𝑡_ −1},then fed into the decoder for the
next timestep. The decoder models the dependencies within the historical generated sequence via a self-attention layer
and integrates visual semantics through a cross-modal attention mechanism to generate the probability distribution
of the current character. This generation process continues until an end-of-sequence token [EOS] is outputted or the
maximum length is reached, forming a complete character sequence _𝑌̂_ = { _̂𝑦_ 1 _, ̂𝑦_ 2 _,_ ... _, ̂𝑦_ 8 }
To maximize performance of MiniLMv2 on a small license plate image dataset, we adopt a transfer learning
approach that combines pre-trained weight initialization with task-specific fine-tuning. We utilize L6xH384 MiniLMv
[17] as the pre-trained model. These weights are obtained through deep training on a large textual dataset and have
been compressed from a larger pre-trained model via knowledge distillation. This method balances rich semantic
modeling with a substantially smaller parameter footprint. This pre-training process endows the model with valuable
prior knowledge of long-range sequence modeling and semantic understanding. Recognizing that complex language
inference is not crucial for license plate recognition, we reduced the architectural complexity by keeping only the first
four Transformer layers of the MiniLMv2.The network parameters of MiniLMv2 are detailed in Table 2.

```
Table 2
Network parameters of MiniLMv
Parameter Name Number of Parameters
Model Layers 4
Hidden Dimensions 128
Number of attention heads 4
FFN hidden layer dimension 512
Parameter quantity 33M
```
#### 3.2. Perspective Transformation Network: PTN

The widely utilized Spatial Transformer Network (STN) [12] commonly performs image rectification by estimating
a 6-parameter affine transformation matrix. Although theoretically extendable to an 8-parameter model for perspective
transformations, direct application of STN for license plate image perspective rectification in practice often yields
noticeably distorted and unnatural results. This is primarily due to the high difficulty in parameter regression, training
instability, and a lack of effective direct supervision (as detailed in Section 5.2).
To address these challenges, this paper proposes a novel Perspective Transformation Network (PTN). Instead
of directly regressing the transformation matrix as in STN, PTN combines the estimation of license plate corner
coordinates with the explicit computation of the perspective transformation matrix. This approach not only effectively
resolves the difficulties STN faces in accurately estimating spatial transformation parameters for perspective-distorted
license plates, but also offers enhanced interpretability, as the estimated corner coordinates can be visualized as needed.
Similar to STN, the proposed PTN could, in principle, learn from supervisory signals provided by a downstream
license plate recognition network. However, given that our chosen recognition network, transLPRNet, inherently
possesses some robustness to variations in capture angle, using its feedback to supervise PTN, while improving
recognition accuracy, often leads to rectification results that are visually inconsistent with human perception. Therefore,


```
Transformation
Solver ModuleMatrix
```
```
Sampling
Module
```
```
Front-view
classification network
```
```
Vertex regression subnetwork
```
```
(Xu1，Xu2)
```
```
(Xv 1 ，Xv 2 )
```
```
Conv Pooling Layer Fully connected layer Feedback
```
```
A
```
```
Figure 5:PTN correction network system block diagram
```
we introduce an innovative training scheme: leveraging weak supervision provided by a dedicated license plate view
classification network to train PTN. This approach significantly simplifies data annotation (where only front-view
license plate images are labeled as positive, and others as negative) and effectively decouples PTN from the recognition
network. Consequently, transLPRNet can be trained with diverse angle license plate images, while during inference,
it only needs to process images rectified by PTN, thereby substantially enhancing recognition accuracy by mitigating
the interference caused by varying capture angles.
At its core, PTN transforms the traditional task of regressing spatial transformation matrix parameters into
the regression of the four corner coordinates (e.g., (x1,y1)...(x4,y4)) of the license plate region. Specifically, PTN
comprises a license plate corner coordinate regression sub-network responsible for estimating these four vertices from
the input image. These estimated coordinates are then fed into a perspective transformation matrix computation module,
which, utilizing the inverse perspective transform formula, derives the complete transformation parameters required
to rectify the license plate from its current distorted pose to a canonical front view. The overall architecture of PTN is
illustrated in Figure 5; its grid generation and sampling modules are adopted from the original STN design.

**_3.2.1. License Plate Vertex Coordinate Regression Sub-network_**
This sub-network is designed to perform the coordinate regression of license plate vertices, with an input image
size of 94×24 pixels. The network first extracts features from the license plate region by employing a multi-layer
convolutional structure to obtain rich local feature information.
Subsequently, two pooling layers are utilized to progressively reduce the spatial dimensions of the feature maps,
thereby enhancing the abstract representation capability of the features. In the deep feature extraction phase, the network
merges and integrates the high-level features through three fully connected layers, ultimately achieving the regression of
the license plate vertex coordinates.Specifically, this sub-network comprises 7 convolutional layers, 2 pooling layers,
and 3 fully connected layers. The convolutional layers are responsible for local spatial feature extraction, while the
pooling layers reduce the spatial dimensions of the features to enhance the model’s robustness to noise. The final
three fully connected layers map the extracted high-level features to eight parameters, which correspond to the two-
dimensional coordinates of the four vertices of the license plate( _𝑋𝑖,𝑌𝑖_ ),where _𝑖_ = 1 _,_ 2 _,_ 3 _,_ 4.
Figure 6 illustrates the architecture of this sub-network. The input is a cropped license plate region image, and
through forward propagation, the network accurately regresses the spatial positions of the four license plate vertices.
This provides essential geometric information for subsequent tasks such as geometric correction and license plate
recognition.Table 3 provides a detailed overview of the network architecture and parameters of the license plate vertex
regression subnet.


```
24 × 94 × 322 × 92 × 3211 ×^46 ×^327 ×^42 ×^327 ×^42 ×^327 ×^42 ×^647 ×^42 ×^1287 ×^42 ×^647 ×^42 ×^32
```
```
Conv
MaxPool
FCL
```
```
896208432
```
```
（X 1 ，Y 1 ）
（X 2 ，Y 2 ）
```
```
（X 3 ，Y 3 ）
```
```
（X 4 ，Y 4 ）
```
```
Figure 6:Vertex regression subnetwork
```
```
Table 3
Architecture and parameters of the license plate vertex regression network
Name Convolution kernel size Convolution stride Input size Output size
Conv 3 × 3 - 24 × 94 × 3 22 × 92 × 32
MaxPool 2 × 2 2 22 × 92 × 32 11 × 46 × 32
Leaky ReLU - - 11 × 46 × 32 11 × 46 × 32
Conv 5 × 5 - 11 × 46 × 32 7 × 42 × 32
Conv 3 × 3 1 7 × 42 × 32 7 × 42 × 32
Conv 3 × 3 1 7 × 42 × 32 7 × 42 × 64
Leaky ReLU - - 7 × 42 × 64 7 × 42 × 64
Conv 3 × 3 1 7 × 42 × 64 7 × 42 × 128
Conv 1 × 1 - 7 × 42 × 128 7 × 42 × 64
Leaky ReLU - - 7 × 42 × 64 7 × 42 × 64
Conv 3 × 3 1 7 × 42 × 64 7 × 42 × 32
MaxPool 3 × 3 3 7 × 42 × 32 2 × 14 × 32
Leaky ReLU - - 2 × 14 × 32 2 × 14 × 32
FCL - - 896 2084
FCL - - 2084 32
FCL - - 32 8
```
**_3.2.2. Perspective Transformation Matrix Estimation Module_**
This module computes the perspective transformation matrix based on the four vertices of the license plate and
the four corner points of the output corrected image.By using the perspective inverse transformation formula [30], the
parameters can ultimately be calculated by solving the following linear equations:

```
⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣
```
##### 𝑋𝑛 1 𝑌𝑛 1 1 0 0 0 − 𝑈𝑚 1 𝑋𝑛 1 − 𝑈𝑚 1 𝑌𝑛 1

##### 0 0 0 𝑋𝑛 1 𝑌𝑛 1 1 − 𝑉𝑚 1 𝑋𝑛 1 − 𝑉𝑚 1 𝑌𝑛 1

##### 𝑋𝑛 2 𝑌𝑛 2 1 0 0 0 − 𝑈𝑚 2 𝑋𝑛 2 − 𝑈𝑚 2 𝑌𝑛 2

##### 0 0 0 𝑋𝑛 2 𝑌𝑛 2 1 − 𝑉𝑚 2 𝑋𝑛 2 − 𝑉𝑚 2 𝑌𝑛 2

##### 𝑋𝑛 3 𝑌𝑛 3 1 0 0 0 − 𝑈𝑚 3 𝑋𝑛 3 − 𝑈𝑚 3 𝑌𝑛 3

##### 0 0 0 𝑋𝑛 3 𝑌𝑛 3 1 − 𝑉𝑚 3 𝑋𝑛 3 − 𝑉𝑚 3 𝑌𝑛 3

##### 𝑋𝑛 4 𝑌𝑛 4 1 0 0 0 − 𝑈𝑚 4 𝑋𝑛 4 − 𝑈𝑚 4 𝑌𝑛 4

##### 0 0 0 𝑋𝑛 4 𝑌𝑛 4 1 − 𝑉𝑚 4 𝑋𝑛 4 − 𝑉𝑚 4 𝑌𝑛 4

##### ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦

##### ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣

##### 𝜃 1

##### 𝜃 4

##### 𝜃 7

##### 𝜃 2

##### 𝜃 5

##### 𝜃 8

##### 𝜃 3

##### 𝜃 6

##### ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦

##### =

##### ⎡ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎢ ⎣

##### 𝑈𝑚 1

##### 𝑉𝑚 1

##### 𝑈𝑚 2

##### 𝑉𝑚 2

##### 𝑈𝑚 3

##### 𝑉𝑚 3

##### 𝑈𝑚 4

##### 𝑉𝑚 4

##### ⎤ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎥ ⎦

##### (2)

Here( _𝑈𝑚𝑖,𝑉𝑚𝑖_ )( _𝑖_ = 1 _,_ 2 _,_ 3 _,_ 4 ) denote the coordinates of the four target points—top-left, top-right, bottom-left, and
bottom-right—on the output license plate image. Meanwhile,( _𝑋𝑛𝑖,𝑌𝑛𝑖_ )( _𝑖_ = 1 _,_ 2 _,_ 3 _,_ 4 ) represent the coordinates of the
corresponding source points within the input license plate image, which also serve as the four corner points of the
desired rectified image. By solving the resulting system of linear equations (as shown in formula 2), eight parameters,
namely _𝜃_ 1 _,𝜃_ 2 _,𝜃_ 3 _,𝜃_ 4 _,𝜃_ 5 _,𝜃_ 6 _,𝜃_ 7 _,𝜃_ 8 are obtained. These parameters constitute the first eight elements of the perspective
inverse transform matrix A in figure 5. The ninth parameter, typically set to 1 as a scaling factor, is appended to


```
(Um 1 ，Vm1) (Um2，Vm2)
```
```
(Um3，Vm3) (Um4，Vm4)
```
```
(Xn 1 ，Yn1)
```
```
(Xn2，Yn2)
```
```
(Xn3，Yn3)
```
```
(Xn4，Yn4)
```
```
𝜃 1 𝜃 2 𝜃 3
𝜃 4 𝜃 5 𝜃 6
𝜃 7 𝜃 8 1
```
```
(Xni,Yni)
```
```
(Uni,Vni)
```
```
A
```
```
Figure 7:Mapping diagram between the LP four vertices and the four corner points of the input or output image
```
complete the matrix. Consequently, the nine parameters are assembled into a 3×3 perspective transformation matrix
A. The mapping relationship between the four vertices of the license plate and the four corner points of either the
original image or the rectified output image is illustrated in Figure 7.

**_3.2.3. Classifier-Guided Weak Supervision for PTN in License Plate Rectification_**
To provide PTN with an independent supervision signal decoupled from high-level recognition networks, this
paper introduces a lightweight MobileNetV3-based [29] frontal license plate classifier after the PTN. This classifier
performs binary classification on license plate images processed by PTN, directly indicating whether the input image
has been successfully rectified into a standard frontal view. This classification result then serves as crucial feedback
for PTN. Considering that the feedback from this classifier is a weak supervision signal relative to PTN’s detailed
geometric rectification task, we propose a two-stage training method. First, the MobileNetV3 frontal license plate
classifier is independently trained using annotated frontal and non-frontal license plate images (as shown in Stage 1 of
Figure 8). Subsequently, after freezing the parameters of this classifier, it is integrated with the PTN model to enable
end-to-end training of the PTN (as shown in Stage 2 of Figure 8). Detailed experimental setups and result analysis will
be elaborated in the experimental section of this paper.

#### PTN: Perspective TranformationNetwork

#### Frontal license plate image classifier

```
(MobileNetv3 based)
```
```
frontal LPs non-frontal LPs
```
```
frontal lp
```
```
non-frontal lp
```
```
Stage 1
```
```
weak
supervisition
signals
```
```
Stage 2
```
```
To TransLPRNet
after being trained well
```
```
Figure 8:illustration of PTN supervision signals from the frontal license plate image classifier
```

#### 3.3. Construction of a Single/Double-Line License Plate Image Dataset

License plate image datasets collected in unconstrained environments should exhibit sample diversity, encompass-
ing a wide range of angles, lighting conditions, and various interferences to more accurately reflect the complexities of
real-world usage. Several publicly available license plate datasets are currently employed in research, such as CLPD
[27] and LSV-LP [28]. While these datasets demonstrate certain effectiveness in specific scenarios, they often suffer
from limitations like insufficient environmental diversity and a limited number of license plate samples.In contrast,
the CCPD [18] dataset is more adaptable for license plate recognition in unconstrained environments. As a publicly
available dataset specifically designed for Chinese license plates, it comprises seven subsets with approximately
280,000 blue license plate images. These images cover a variety of scenes, angles, lighting conditions, and weather.
The dataset provides detailed annotation information, including license plate locations and numbers.
Despite being the most widely used public dataset for research in license plate recognition in unconstrained
environments, the CCPD dataset still has inherent issues. First, its annotation employs an iterative strategy, where a
model is trained on a portion of the annotated data, then used to predict the vertices and bounding boxes of the license
plates. This process can introduce annotation errors. Second, the dataset only contains single-line license plates, lacking
samples of double-line plates. Furthermore, although the subsets cover various scenes, the distribution of license plate
samples across different scenes is imbalanced. Some scenes have abundant samples, while others are severely lacking,
which limits its potential applications.To address these problems, this paper proposes two improvements: First, we
correct the mislabeled license plates. Second, we introduce a strategy to integrate multi-type license plates (single-
line and double-line), by pasting synthetic double-line license plates onto redundant single-line license plate images;
thereby building a more diverse and comprehensive license plate dataset, which enables improving robustness and
generalization capabilities for LPR algorithms.

**_3.3.1. Preprocessing of License Plate Image Dataset Used in the Experiments_**
This paper focuses on license plate image correction and recognition, and does not involve the localization
process. To evaluate the impact of real-world license plate localization errors on subsequent correction and recognition
performance, we utilize the bounding boxes and four-vertex coordinate information provided by the CCPD dataset,
and apply random perturbations on top to simulate the potential effects of actual localization inaccuracies on the
license plate images.During the experiments, we found that the original CCPD dataset contains annotation errors,
mainly because its labels are generated through a combination of manual annotation and model predictions, which
can inevitably introduce inaccuracies. As shown in Figure 9, mapping images using the original CCPD labels results
in noticeable misalignments in some cases; in contrast, Figure 10 illustrates that after our systematic correction, the
label mappings are more accurate and properly aligned.Therefore, directly using uncorrected labels for image mapping
and applying random disturbances could significantly affect the accuracy of subsequent license plate correction and
recognition. To ensure the reliability and validity of the experimental results, we performed a systematic correction of
the label errors in the CCPD dataset before conducting further experiments.

```
Figure 9:Illustration of license plate images before label correction
```
This paper employs the license plate detection model used in the CCPD study [18] to detect license plates in the
images. Subsequently, the Intersection Over Union (IOU) between the predicted bounding boxes and the annotated
ground-truth bounding boxes is calculated. If the IOU exceeds 0.6, the annotation is considered correct; otherwise, it
is deemed to contain an error. As illustrated in Figure 11, among the filtered samples identified as having incorrect
annotations, the red boxes indicate the model’s predicted license plate locations, while the green boxes represent the


```
Figure 10:Illustration of license plate images after label correction
```
original ground-truth annotations. Through this filtering process, a total of 1,414 images with annotation errors were
identified. For these problematic samples, manual re-annotation was performed to correct their labels, resulting in
a refined, accurately annotated license plate dataset that provides a more reliable foundation for subsequent model
training.

Figure 11:Illustration of detection results versus original labels, where the red boxes indicate the model’s predicted license
plate locations, and the green boxes represent the original annotated license plate locations

**_3.3.2. Double/Single-Line License Plate Dataset Construction via Image Overlay_**
Addressing the challenge of limited double-line license plate datasets, which significantly hampers the training
effectiveness of the prposed TransLPRNet model, this paper proposes an image overlay approach. By retaining the
original CCPD single-line license plate images, we effectively compress and leverage existing data resources while
simultaneously enhancing the diversity of double-line license plate samples. Specifically, without expanding the
original image set’s size, we overlay synthesized double-line license plate images onto pre-existing, correctly identified
single-line license plate images from the CCPD dataset, enabling the fusion and augmentation of double/single-line
license plate data. This scheme, requiring neither additional data acquisition costs nor an increased number of images,
preserves the rich information within the original data while successfully introducing double-line license plate samples.
This significantly enhances the training set’s diversity and generalization capabilities.
The specific procedure is as follows: First, the “base” subset of the CCPD dataset is randomly and evenly divided
into two parts: one for model training (100,000 images) and the other for model testing (100,000 images). Given
that the original CCPD authors employed RPNet [18] for license plate detection and recognition in their paper, we
adopt the RPNet model as a baseline for training and performance evaluation. Since these license plate images can be
correctly recognized by models trained on the training set, they are considered redundant relative to the entire base
dataset. Therefore, these correctly recognized plates can serve as redundant samples for subsequent texture mapping.
To enhance the realism of the synthetic double-line license plates, we first apply a blurring process to these images,
generating blurred samples of yellow and green double-line license plates (25,000 of each color), simulating image
blur and environmental variations encountered in real-world scenarios. Next, using image overlay techniques, these
synthesized blurred double-line license plate images are superimposed onto the redundant single-line license plate
samples, forming a composite training set containing multiple license plate types. This method, without increasing the
original image set’s size, enables effective data compression, thorough utilization of information, and the introduction
of diverse categories, ultimately constructing a multi-type license plate training set with enhanced generalization
capabilities. Further details regarding the construction process can be found in Figure 12.
Building upon the existing CCPD’s original test set, this paper introduces a dedicated double-line license plate test
set. This set was created by superimposing double-line license plate images onto various subsets of the CCPD dataset
and subsequently extracting image crops along with their perturbed label information. In total, the set comprises 80,


```
Overlaying
```
```
Base
```
```
Training set
```
```
Test set
```
```
Identify the wrong
```
```
Identify the right
```
```
Multi-type license plate dataset
```
```
Figure 12:Flowchart of composite training dataset construction
```
license plate images, encompassing both yellow and green double-line variants. Figure 13 shows example cropped
images from the double-line license plate test set.The training set, constructed by compressing redundant images and
applying augmentation to the "base" subset of the CCPD dataset, contains both single-line and double-line license
plates. The data distribution statistics for the single-line and double-line license plate test sets are shown in Figure

16. To ensure dataset diversity and enhance the generalizability of the recognition model, the double-line license
plate images used for augmentation were carefully selected to maintain uniform distributions with respect to both
quantity and geographic origin, effectively mitigating the risk of overfitting to frequently occurring provinces such as
Anhui (“wan”). To illustrate the impact of augmentation on data distribution, Figure 14 presents a histogram depicting
the provincial distribution of license plates within the initial training set, while Figure 15 shows the corresponding
distribution after augmentation. Figures 17 and 18 illustrate the generated double-line license plates after blurring and
perspective warping, superimposed onto the license plate region of the vehicle in the CCPD dataset.

```
Figure 13:Cropped image example of double-line license plate test set
```
### 4. Experiment settings and result analysis

All experiments were conducted within the following environment: operating system Ubuntu 18.04, deep learning
framework PyTorch 1.8 with CUDA 11.1, and hardware consisting of a TITAN X GPU, an Intel Xeon E5-2620 v
CPU, and80GB of memory.

#### 4.1. Simulation of License Plate Detection Errors Based on Random Coordinate Perturbations

To simulate potential errors inherent in license plate detection, we introduce random coordinate perturbations to
meticulously calibrated license plate image labels. This perturbation method involves adding a Gaussian-distributed
random offset, characterized by a mean of 0 and a standard deviation of 4, to the original label coordinates. Specifically,


```
皖苏浙沪豫粤京鄂鲁闽冀赣川湘渝晋津辽陕蒙黑甘云贵新桂吉青琼宁藏
```
```
Figure 14:Histogram of provincial distribution prior to training set augmentation
```
```
皖苏浙沪豫粤京鄂鲁闽冀赣川湘渝晋津辽陕蒙黑甘云贵新桂吉青琼宁藏
```
```
Figure 15:Histogram of provincial distribution following training set augmentation
```
```
Double/Single-line license plate train set
```
```
Weather（10k）
Challenge（10k）
```
```
（20kFn）
（Db20k）
```
```
（Rotate10k）
（Tilt10k）
```
```
（Base25k-d）
Blue single line（60k）
```
```
Green（ double line10k）
Yellow double line（10k）
```
```
（Tilt10k）
```
```
（Fn20k）
```
```
（Db20k）
```
```
（Base75k-s）
```
```
Double-line license plate test set Single-line^ license plate^ test set
```
```
Figure 16:Distribution pie charts of single-line and double-line license plate training and test sets
```
for bounding box-based detection, the coordinates of the top-left and bottom-right corner points are independently
perturbed(Bounding box localization disturbance). In contrast, for fine-grained localization utilizing license plate
vertices, random deviations are applied to the coordinates of all four corner points(Four Vertex location disturbance).
This approach aims to effectively mimic realistic error scenarios encountered during practical license plate detection,
as visually demonstrated in Figure 19, and enhances the robustness and generalization capabilities of the model.


```
Generated green double-
line license plate
```
```
Green double-line license
plate blurred
```
```
Generated yellow double-
line license plate
```
```
Yellow double-line license
plate blurred
```
```
Figure 17:Illustration of a double-line license plate
```
```
CCPD2019 Multi-type license plate
training set
```
```
Multi-type license plate
test set
```
```
Figure 18:Examples of generated double-line license plates overlaid on images from the CCPD dataset
```
```
(a) Bounding box localization
```
```
(b) Bounding box localization disturbance
```
```
(d) Four Vertex location disturbance
```
```
(c) Four Vertex location
```
```
Figure 19:Illustration of coarse and fine perturbations in license plate localization
```
#### 4.2. PTN Training and Evaluation

To further enhance the convergence stability of the license plate correction module based on weakly supervised
classification information during training, a three-stage training strategy was employed. In the first stage, the correction
network (PTN) was frozen, and only the classification network was trained to ensure robust feature extraction
capabilities. In the second stage, the classification network was frozen, and the correction network (PTN) was trained
using a randomized mixture of positive and negative samples to prevent alterations in feature extraction patterns. In the
third stage, the trained correction network was combined with the frozen classification network, and the entire model


was fine-tuned through end-to-end training, leveraging feedback from the classification output to iteratively optimize
the correction performance of PTN. Throughout the training process, a relatively low learning rate was adopted to
enable fine-grained adjustments of the network parameters, thereby effectively improving overall model performance.
The hyperparameter configuration used in this experiment is detailed in Table 4, and the training process is illustrated
in Figure 20.

## PTN ClassificationNetwork

```
Training 1 Lr=0.
```
```
PTN ClassificationNetwork
```
```
Training 2 Lr=0.
Feedback
```
## 

```
PTN ClassificationNetwork
```
```
Training 3 Lr=0.
Feedback
```
##  

## Train

```
Freeze
```
```
Figure 20:Training workflow of the PTN
```
```
Table 4
Hyperparameter configuration for PTN training
Parameter Name Number of Parameters
Learning Rate 0.
Optimizer Adam
Batch Size 32
Training Epochs 100
Image Size 94 × 24
```
In theory, the Spatial Transformer Network (STN) [12] is capable of learning parameters for the spatial ge-
ometric transformation of images. However, the STN is primarily designed for spatial correction related to affine
transformations, such as image translation, rotation, and scaling. Directly employing an STN to regress a perspective
transformation matrix often leads to convergence challenges during network training. This paper presents a preliminary
investigation into using STNs to regress both a 6-parameter affine transformation matrix and an 8-parameter perspective
transformation matrix. Results of these experiments are shown in Figures 21 and 22. The experimental results indicate
that the STN provides limited correction of car license plate tilt when regressing affine transformations. Conversely,
when the STN is used to regress perspective transformation parameters, the output images exhibit large areas of black,
rendering complete license plate content unrecognizable. This phenomenon is primarily attributable to the high degree
of interdependence between parameters in the perspective transformation matrix. Slight variations in any parameter
within the perspective transformation can induce complex and non-linear geometric distortions in the output image,
leading to significant stretching or compression of the license plate image, and ultimately resulting in substantial invalid
regions in the transformed image.Figure 23 shows a comparison of images before and after PTN correction proposed
in this paper.


```
Figure 21:Example of STN before and after affine transformation correction
```
```
Figure 22:Example images before and after STN perspective transformation correction
```
```
(a) Before PTN correction
```
```
(b) After PTN correction
```
```
(a) Before correction
```
```
(b) After correction
```
```
Figure 23:Comparison of images before and after PTN correction
```
This paper also investigates integrating a Perspective Transformation Network (PTN) with TransLPRNet, lever-
aging TransLPRNet to provide feedback to the PTN. The corresponding experimental results are shown in Figure

24. The results indicate that the integration of PTN and TransLPRNet can potentially lead to suboptimal rectification
performance, and may even exacerbate the geometric distortion of license plate images. This is primarily due to the fact
that TransLPRNet is capable of processing license plates with a certain degree of tilt; its performance degrades only
under conditions of extreme tilt. Consequently, when TransLPRNet can still effectively recognize the license plate, the
PTN struggles to obtain effective feedback adjustment signals, and thus fails to spatially correct the license plate image
accurately.

```
(a) Before PTN correction
```
```
(b) After PTN correction
```
```
(a) Before correction
```
```
(b) After correction
```
```
Figure 24:Experimental results of PTN integrated with TransLPRNet
```

#### 4.3. TransLPRNet Training and Evaluation

To ensure fairness in the subsequent comparative experiments, the TransLPRNet model in this study was trained
using the recommended data partitioning strategy from the CCPD dataset [18]. The newly constructed multi-type
license plate dataset was randomly split into two equal parts—one used for training and validation (with an 8:2 ratio),
and the other used for testing. Unlike the original base test set in CCPD, this test set also includes double-line license
plates. Therefore, in this study, we further divided it into two subsets: base-s (single-line license plate test set) and
base-d (double-line license plate test set).Although the number of samples in the extended single-line and double-line
license plate dataset matches that of the original CCPD base training and test sets, the number of single-line plates in
both the training and test sets is lower due to redundancy compression and the addition of double-line plates. Despite
the reduced quantity, the dataset retains maximal informational value through redundancy compression. As a result, the
recognition performance of the trained license plate model remains comparable to that trained on the original CCPD
base set. Detailed results are presented in the comparative experiments in Section 5.3.1, and the hyperparameter settings
used for training are listed in Table 5.

```
Table 5
Hyperparameter configuration for TransLPRNet training
Parameter Name Number of Parameters
Learning Rate 0.
Optimizer Adam
Batch size 64
Training Epochs 200
Image size 224 × 224
```
To enhance the model’s adaptability in open-world scenarios, this paper incorporates various data augmentation
methods into the TransLPRNet model training, thereby expanding the diversity of the training dataset and improving the
model’s robustness against different scenes and interference factors. The augmentation strategy includes the following
techniques: random cropping to simulate variations in license plate position and size; random rotation of license plates
to accommodate different viewing angles; color jittering to emulate changes in lighting conditions; random perspective
transformation of license plates to mimic camera viewpoint deviations; and random erasing to simulate license plate
occlusion or noise. The combination of these data augmentation techniques effectively enriches the training samples,
ultimately improving the model’s license plate recognition performance in real-world applications. The specific effects
are illustrated in Figure 25.

```
Figure 25:Data augmentation visualization
```
**_4.3.1. Recognition results on the CCPD test Dataset_**
To verify the effectiveness of the proposed TransLPRNet algorithm, a series of comparative experiments were
conducted. Given that the dataset used in this study is a corrected version of the CCPD license plate dataset,


mainstream and open-source license plate recognition algorithms were selected and re-implemented under the same
dataset conditions to ensure a fair comparison.For methods described in references [31][32][33][34], which do not
provide publicly available code, the experimental results reported in their original papers were used for comparison.
Specifically, references [31],[33] and [34] adopt rectangular bounding box localization methods, while reference
[32] uses vertex-based license plate localization.To minimize the influence of localization methods on recognition
performance, this study employed random perturbation strategies based on rectangular bounding boxes and license
plate vertex localization to extract license plate regions. A comprehensive comparison was made on the corrected
CCPD dataset, evaluating recognition accuracy across different subsets, mean recognition accuracy, inference speed,
and model size for each algorithm using various localization approaches. The experimental results are presented in
Tables 6 and 7.

```
Table 6
Performance comparison on the corrected CCPD test set under coarse localization conditions
Recognition rate/% Avg Base-s Db Fn Rotate Tilt Weather Challenge Size(MB) FPS
LPRNet [8] 90.98 96.47 90.21 88.67 74.31 83.97 89.48 67.49 1.8 3072
RPNet [18] 92.49 97.23 91.82 90.21 85.98 83.49 83.87 75.04 210 61
Eulpr [19] 97.10 99.09 97.39 94.56 95.47 96.76 96.35 82.67 3.9 1547
PaddleOCRv3 [21] 97.51 98.69 98.12 95.02 96.31 97.14 96.89 91.65 12.8 231
TrOCR [15] 98.82 99.23 99.14 99.27 99.15 99.27 98.97 92.18 138.7 32
PDLPR [31] 99.4 99.9 99.5 99.5 99.5 99.3 99.4 94.1 - 159.8
OFANet [33] 99.41 99.9 99.4 99.4 94.6 99.6 99 93.8 - -
MP-LPR [34] 99.24 99.68 98.42 98.54 99.1 98.85 99.39 98.34 - -
TransLPRNet 99.34 99.58 99.54 99.60 99.64 99.75 99.41 95.21 32.8 167
```
```
Table 7
Performance comparison on the corrected CCPD test set under fine localization conditions
Recognition rate/% Avg Base-s Db Fn Rotate Tilt Weather Challenge Size(MB) FPS
LPRNet [8] 98.30 99.44 98.24 98.58 98.71 98.74 96.98 86.97 1.8 3072
RPNet [18] 95.54 98.37 96.74 94.98 90.21 92.41 86.57 83.47 210 61
Eulpr [19] 98.57 99.48 98.75 98.69 99.07 99.09 97.74 88.68 3.9 1547
PaddleOCRv3 [21] 98.99 99.65 98.82 98.91 98.78 99.01 99.07 92.93 12.8 231
TrOCR [15] 99.32 99.69 99.39 99.51 99.75 99.67 99.32 94.37 138.7 32
Liu et al.(2024) [32] 99.32 99.78 99.49 99.45 99.32 99.63 99.29 94.25 - -
TransLPRNet 99.58 99.74 99.65 99.74 99.85 99.84 99.59 96.98 32.8 167
```
As shown in Tables 6 and 7, TransLPRNet achieves the highest recognition accuracy on most test subsets for license
plate images obtained through both coarse localization (Bounding box localization) and fine localization (Four Vertex
location) perturbations. The only exceptions are: on the Challenge subset under coarse localization, its accuracy is
slightly lower than the method proposed in [34]; on the Base-s subset, it is slightly lower than the algorithms from
[31], [33], and [34]; and under fine localization on the Base-s subset, it is slightly outperformed by the method in
[32].One contributing factor to this outcome is that the base training set used in this study contains fewer single-line
license plate samples compared to the datasets used in [31]–[34]. Although redundancy compression was applied to
maximize information retention from the original CCPD base dataset, a slight impact remains. Another important
reason lies in the localization methods themselves: [31], [33], and [34] used rectangular bounding box localization,
while [32] adopted vertex-based localization. Due to inherent localization errors, these methods may automatically
filter out samples that are difficult to localize, thereby achieving higher recognition accuracy during testing.
Additionally, [31] and [33] adopt a unified detection and recognition framework, where only detection results with
an IOU greater than 0.7 are passed to the recognition module. This implies that samples with poor detection quality or
difficult localization may be automatically excluded during testing, which can lead to inflated recognition accuracy on
certain subsets, such as the Base-s subset. In contrast, our experiments used the complete test set without filtering any
samples, making the evaluation results more objective. If [31]–[34] were evaluated on the full test set, their recognition


accuracy might decrease.Furthermore, thanks to the integration of a visual encoder and text decoder, TransLPRNet
exhibits more stable recognition performance under varying localization perturbation conditions compared to other
algorithms such as LPRNet, PRNet, Eulpr, PaddleOCRv3, and PDLRP. In terms of inference speed, apart from the
optimized PaddleOCRv3 and pure CNN-based models like LPRNet and its improved version Eulpr, TransLPRNet also
demonstrates competitive computational efficiency. Figure 26 presents a visualization of the single-line license plate
recognition results.

```
Label : 皖AEE925
Predict: 皖AEE925
Label : 鄂A9XT19
Predict: 鄂A9XT19
Label : 京LD7865
Predict: 京LD7865
Label : 皖AS5596
Predict: 皖AS5596
```
```
Label : 皖AD4830
Predict: 皖AD4830
```
```
Label : 皖AC808P
Predict: 皖AC808P
```
```
Label : 皖A68G72
Predict: 皖A68G72
```
```
Label : 皖A3S695
Predict: 皖A3S695
```
```
Figure 26:Visualization of single-line license plate image recognition results
```
**_4.3.2. License Plate Recognition Results on the Extended double-line Test Dataset_**
To further verify the effectiveness of the proposed algorithm in double-line license plate recognition, relevant
experiments were conducted on a custom-built dataset of double-line license plates. As some of the algorithms
compared in the previous section do not support double-line license plate recognition, their recognition accuracy in this
experiment is marked as “×”. Although [33] and [32] support double-line license plates, their recognition results are
marked as “–” due to the lack of publicly available code. References [31] and [34] neither support double-line license
plate recognition nor provide publicly available code, and thus are excluded from comparison in this part. Detailed
experimental results are shown in Table 8.

```
Table 8
Recognition accuracy on the expanded double-line license plate data test set
Recognition rate/% Avg Base-d Db Fn Rotate Tilt Weather Challenge
LPRNet [8] ××××××××
RPNet [18] ××××××××
OFANet [33] - - - - - - - -
Liu et al.(2024) [32] - - - - - - - -
Eulpr [19] 86.43 87.21 87.37 85.19 83.98 84.36 86.29 83.92
PaddleOCRv3 [21] 96.27 97.23 97.19 94.87 94.92 94.29 95.43 91.74
TrOCR [15] 98.13 98.03 99.48 99.63 99.14 99.27 99.85 96.47
TransLPRNet 98.70 98.57 99.01 99.13 99.17 99.38 99.05 97.07
```
As shown in Table 8, Eulpr, PaddleOCRv3, TrOCR, and the proposed algorithm are all capable of recognizing
double-line license plates. Among these, our method consistently outperforms the others across all subsets, achieving
the best recognition performance. Notably, in the Fn, Tilt, and Challenge subsets, our algorithm attained accuracy rates
of 99.13%, 99.38%, and 97.07%, respectively, significantly surpassing other approaches. These results demonstrate
the robustness and practicality of the proposed method for double-line license plate recognition.Figure 27 shows the
visualization of double-line license plate recognition results.
Combining the results from Table 6,Table 7 and Table 8, it can be concluded that after data augmentation, the
model not only maintains the recognition accuracy on the original CCPD test set but also exhibits high accuracy on the


Table 9
Model performance ablation experiment results based on pre-trained weights

```
Experiment Encoder Pre-trained Decoder Pre-trained Single-line Accuracy Double-line Accuracy
Experiment (A) ×× 75.21 85.14
Experiment (B) ✓ × 97.15 97.82
Experiment (C) × ✓ 94.14 95.87
Experiment (D) ✓✓ 99.34 98.70
```
double-line license plate test set. This is primarily due to retaining the core information within the base subset, which
prevents the loss of key features caused by data replacement. By replacing redundant images, the approach effectively
reduces the impact of duplicate samples within the dataset, thereby lowering the risk of overfitting. Furthermore,
incorporating double-line license plate images enhances the model’s generalization capabilities.Additionally, since the
dataset augmentation was performed without increasing the number of samples in the base subset, the training time of
the TransLPRNet model remains nearly unaffected. This achieves a dual optimization of data augmentation efficacy
and training efficiency.

```
Label : 蒙XR6CN3
Predict: 蒙XR6CN3
```
```
Label : 藏CZ952T
Predict: 藏CZ952T
```
```
Label : 川B20H6B
Predict: 川B20H6B
```
```
Label : 沪MAT98U
Predict: 沪MAT98U
```
```
Label : 川32S3K8X
Predict: 川32S3K8X
```
```
Label : 陕08KPKA8
Predict: 陕08KPKA8
```
```
Label : 甘37Y4Y37
Predict: 甘37Y4Y37
```
```
Label : 辽68JAX2K
Predict: 辽68JAX2K
```
```
Figure 27:Visualization of double-line license plate image recognition results
```
#### 4.4. Ablation experiments

To systematically evaluate the contribution and effectiveness of the key modules proposed in this paper, two sets of
ablation experiments were conducted. The first set aimed to assess the critical role of pre-trained weights in improving
model performance, while the second set focused on validating the effectiveness and optimization benefits of the
proposed PTN correction algorithm.

**_4.4.1. Ablation Experiments on the Impact of Pre-trained Weights_**
To enhance the model’s recognition performance and convergence speed, this paper utilizes pre-trained Mobile-
ViTv3 [16] and MiniLMv2 [17] weights to initialize the encoder and decoder. To validate the specific contribution of
these pre-trained weights to model performance, we designed a systematic ablation study. This study comprises four
distinct configurations: Group A represents the scenario where neither the encoder nor the decoder employs pre-trained
weights; Group B uses the pre-trained weights for the encoder but not for the decoder; Group C uses the pre-trained
weights for the decoder but not for the encoder; and finally, Group D employs pre-trained weights for both the encoder
and the decoder. Under identical training data and experimental conditions, the experimental results are shown in Table
9, allowing for a comparative analysis of the performance differences across each configuration.
The experimental results demonstrate that the model achieves the highest accuracy in both single-line and double-
line license plate recognition tasks when pre-trained weights are applied to both the encoder and decoder. In contrast,
using only partial pre-trained weights leads to an approximate 4% age point drop in recognition accuracy, while
completely omitting pre-trained weights results in a decline of about 20% age points. These findings indicate that


pre-trained weights significantly enhance the model’s recognition performance, and that the performance improvement
becomes more pronounced as more pre-trained weights are incorporated. This underscores the importance of effectively
leveraging pre-trained weights to improve the model’s recognition capability.

**_4.4.2. Evaluating the Effect of PTN through Ablation Experiments_**
To validate the practical effectiveness of the proposed license plate rectification algorithm PTN in enhancing
model recognition performance, this study designed four comparative experiments. Specifically, the first experiment
involved recognition after applying perturbations to coarsely localized license plates. The second experiment involved
recognition after applying PTN rectification to the coarsely perturbed license plates. The third experiment involved
recognition following perturbations to finely localized license plates, while the fourth experiment involved recognition
after applying PTN rectification to the finely perturbed license plates. To ensure the fairness of the comparisons,
all experiments employed the same training strategy and training data, with different localization and rectification
methods introduced only during the testing phase. The test set includes corrected single-line and double-line license
plate samples with previously mislabeled annotations. The detailed comparison results are shown in Table 10 and
Table11.

```
Table 10
Comparison of recognition accuracy on single-line license plate test set with different localization methods and
PTN correction
Experiment Avg Base-s Db Fn Rotate Tilt Weather Challenge
Bounding box localization disturbance 99.34 99.58 99.54 99.60 99.64 99.75 99.41 95.21
Bounding box localization disturbance+PTN 99.63 99.76 99.70 99.77 99.87 99.86 99.62 97.47
Four Vertex location disturbance 99.58 99.74 99.65 99.74 99.85 99.84 99.59 96.98
Four Vertex location disturbance+PTN 99.66 99.79 99.72 99.80 99.90 99.91 99.67 97.61
```
```
Table 11
Comparison of recognition accuracy on double-line license plate test set with different localization methods
and PTN Correction
Experiment Avg Base-d Db Fn Rotate Tilt Weather Challenge
Bounding box localization disturbance 98.70 98.57 99.01 99.13 99.17 99.38 99.05 97.07
Bounding box localization disturbance+PTN 98.87 98.79 99.15 99.24 99.26 99.41 99.11 97.26
Four Vertex location disturbance 98.74 98.61 99.04 99.16 99.18 99.39 99.07 97.11
Four Vertex location disturbance+PTN 98.92 98.85 99.18 99.27 99.30 99.42 99.15 97.37
```
```
Before correction
```
```
After correction
```
```
Recognition result
```
```
赣AZJ234
```
```
皖AZJ234
```
```
皖AP5Z2J
```
```
皖AP522J
```
```
湘D38K4G9
```
```
湘038K4G9
```
```
吉 2 P3Z53
```
```
吉ZP3Z53
```
```
Recognition result
```
```
Label 皖AZJ234 皖AP522J 湘038K4G9 吉ZP3Z53
```
```
Label 皖AZJ234 皖AP522J 湘038K4G9 吉ZP3Z53
```
```
Figure 28:Comparison of recognition results with and without using PTN
```
Based on the data presented in Tables 8 and 9, the proposed PTN-based correction algorithm effectively improves
recognition accuracy across different license plate localization methods and diverse scenarios. Whether applied to


```
Figure 29:Distribution of recognition errors by error type
```
single-line or double-line license plate test sets, the integration of the PTN correction module significantly enhances
recognition performance across various subsets.For instance, applying PTN correction on single-line license plates
with Bounding box localization disturbances increases the recognition accuracy on the challenge subset from 95.21%
to 97.47%. Similarly, for double-line license plates under Bounding box localization disturbances, accuracy improves
from 97.07% to 97.26% after PTN correction. PTN also demonstrates strong performance on other subsets, indicating
its effectiveness in addressing recognition difficulties caused by excessive license plate tilt.To further validate the
effectiveness of PTN, several scene examples were selected for visual analysis, as shown in Figure 28. The license plate
images without PTN correction exhibit noticeable skew and distortion, which lead to recognition errors. In contrast,
after applying PTN correction, the character alignment becomes more regular and clear, thereby significantly improving
the recognition model’s accuracy and robustness.
This study analyzed the corrected CCPD test set and identified a total of 647 license plate images with recognition
errors, as shown in Figure 29. These misrecognized license plates can be broadly categorized into two types.The
first category of errors stems from the high morphological similarity between certain characters, accounting for 254
images. Specifically, confusions frequently occur between digits and letters such as “8” and “B,” “2” and “Z,” “0”
and “D,” as well as “5” and “S,” as illustrated in Figure 30. These recognition errors are primarily due to the inherent
structural similarities among these characters. For example, “8” and “B” or “D” and “0” share overlapping contour
features.Additionally, license plate images captured in open environments often suffer from limitations such as low
resolution, uneven lighting, and character distortion. These factors can lead to the loss of critical distinguishing features
in the characters, further contributing to recognition errors.
Another factor contributing to license plate recognition errors is the high degree of blurriness inherent to
certain license plate images. Such blurriness exceeds the capacity of the human visual system to accurately discern
individual characters, resulting in the critical features of each character being substantially compromised. Image
quality degradation leads to the loss of fine details in the characters, and even after effective correction using PTN,
misrecognition during subsequent processing stages remains a significant challenge. Additionally, for these highly
blurred license plates, the Chinese character segment is frequently predicted as “wan,” primarily due to the fact that
the CCPD dataset predominantly contains license plates from Anhui Province. This bias during training causes the
model to overly associate such characters with “wan.” While this study has attempted to address potential imbalance
by augmenting the dataset to improve the distribution of double-line license plates, it has not sufficiently balanced
the provincial distribution, which adversely affects the model’s generalization capability. A total of 372 such blurred
license plates were identified, with representative examples provided in Figure 31.


```
Label : 皖AD515W
Predict: 皖A 0 515W
```
```
Label : 皖AZ8355
Predict: 皖A 28355
```
```
Label : 皖A59H88
Predict: 皖A59HB 8
```
```
Label : 皖AT319S
Predict: 皖AT319 5
```
```
Label : 皖AD601Z
Predict: 皖A 0 601Z
```
```
Label : 皖A592T6
Predict: 皖A59ZT6
```
```
Label : 皖AA5B85
Predict: 皖AA5 885
Label : 皖ASS556
Predict: 皖AS 5556
```
```
‘ 0 ’ confused with ‘D’ ‘ 2 ’ confused with ‘Z’ ‘ 8 ’ confused with ‘B’ ‘ 5 ’ confused with ‘S’
```
```
Figure 30:Illustration of license plate character confusion and recognition errors
```
```
皖A5K 223
```
```
皖K4 1273
```
```
皖A 4 F 605
```
```
皖AH 1305
```
```
皖A 8525 H
```
```
皖A 85231
```
```
皖A79Z 76
```
```
皖AZ0 276
```
```
Recognition result
```
```
Before correction
```
```
After correction
```
```
Recognition result
```
```
Label 皖B21273 皖AHF305 皖AB523H 皖A7P276
```
```
Label 皖B21273 皖AHF305 皖AB523H 皖A7P276
```
```
Figure 31:Illustration of license plate character confusion and recognition errors
```
### 5. Conclusion and Outlook

This paper introduces TransLPRNet, a novel license plate recognition network that integrates a lightweight, pre-
trained visual encoder and a text decoder. By employing a Transformer architecture to globally model the inter-
relationships between license plate image patches, TransLPRNet effectively mitigates the character loss or spurious
character addition issues commonly encountered in CNN+CTC and CNN+RNN based license plate recognition
algorithms. Furthermore, in conjunction with the proposed Perspective Transformation Network (PTN), TransLPRNet
achieves accurate and rapid recognition of single-line and double-line Chinese license plates in a variety of complex
scenarios.
Beyond providing training supervision for the PTN, the proposed front-view license plate classification network
also exhibits potential for license plate image quality assessment and fine-grained license plate type classification. This
information can be leveraged to enhance the confidence estimation of license plate recognition results, aiding in the
determination of result correctness. In future work, we plan to explore the adoption of a Transformer encoder-only
architecture in place of the current Transformer encoder-decoder structure within TransLPRNet, with the potential to
further improve inference speed.


### CRediT authorship contribution statement

**Guangzhu Xu:** Conceptualization of this study, Methodology development, Designing the experiments, Critical
review and editing of the manuscript, Writing – Original Draft Preparation (Primary and Core Sections). **Zhi Ke:** Data
curation and processing, Conducting formal analysis, Contributing to the methodology and software development,
Visualization of results. Writing – Original Draft Preparation (Significant Sections). **Pengcheng Zuo:** Data curation
and processing, Validating the experimental results, Contributing to software implementation. **Bangjun Lei:** Project
administration, Critical review and editing of the manuscript, Securing funding for the research.