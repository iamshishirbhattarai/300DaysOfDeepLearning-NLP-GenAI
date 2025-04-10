# 300DaysOfDeepLearning-NLP-GenAI
This is my 300 days of journey from Deep Learning to Generative AI !!


___
## Syllabus to cover


| **S.N.** | **Books and Lessons (Resources)**                                                                                         | **Status** |
|----------|---------------------------------------------------------------------------------------------------------------------------|------------| 
| **1.**   | [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)                                | ✅          |
| **2.**   | [**NLP Specialization**](https://www.coursera.org/specializations/natural-language-processing)                            | ✅          |  
| **3.**   | [**LLM-Course Repo**](https://github.com/mlabonne/llm-course/tree/main)                                                   | ⏳          |
| **4.**   | [**Building Agentic RAG with LlamaIndex**](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)| ✅  |


## Research Papers

| **S.N.** | **Papers**                                                                                                                                                                   | 
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1.**   | [**Learning representations by back-propagating errors**](https://www.nature.com/articles/323533a0)                                                                          |
| **2.**   | [**ImageNet Classification with Deep Convolutional Neural Networks**](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| **3.**   | [**You Only Look Once: Unified, Real-Time Object Detection**](https://arxiv.org/abs/1506.02640)                                                                              |
| **4.**   | [**Neural Machine Translation By Jointly Learning To Align And Translate**](https://arxiv.org/pdf/1409.0473)                                                                 |
| **5.**   | [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762)                                                                                                            |
| **6.**   | [**Large Language Models as Data Preprocessors**](https://arxiv.org/pdf/2308.16361)                                                                                          |
| **7.**   | [**Late Chunking: Contextual Chunk Embeddings  Using Long-Context Embedding Models**](https://arxiv.org/pdf/2409.04701)                                                      |

## Projects

| **S.N.** | **Project Title**                                                                                                                        |
|----------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1.       | [**Brain Tumor Detection System**](https://github.com/iamshishirbhattarai/Machine-Learning/tree/main/Brain%20Tumor%20Detection%20System) | 
| 2.       | [**Pneumonia Prediction Using CNN**](https://github.com/iamshishirbhattarai/Pneumonia-Prediction-Using-CNN)                              |
| 3.       | [**Fine Tuning yolov8**](https://github.com/iamshishirbhattarai/Fine-Tuning-YOLOV8)                                                      | 
| 4.       | [**Email Detection Using Naive Bayes Classifier**](https://github.com/iamshishirbhattarai/Email-Spam-Detection)                          |
___

## Day 1

While I had started the **Deep Learning Specialization** few days ago, today I completed the first course of the specialization
**Neural Networks and Deep Learning**. With this confidence being alive, I thought to start my challenge. 
In the course, the specific focus was on the **vectorization** technique that has saved
a lot of time while executing **Neural Networks**. Also, the concept of **Broadcasting** in Python, **Activation Function**, **Cost Function** and 
**Gradient Descent** was provided which was kind of revision to me. Learnt about the mathematics behind **Forward Propagation** & 
**Backward Propagation** through the course. Also took the help of classic research paper [**Learning representations by back-propagating errors**](https://www.nature.com/articles/323533a0)
to understand back-propagation stuffs.

- Some slides snippets from the course are provided below : <br> <br>
  ![Vectorization](Day1_To_10/day1_vectorization.png) <br> <br> ![gradient_descent](Day1_To_10/day1_gradientDescent.png) <br> <br>
  ![forward_backward_props](Day1_To_10/day1_forwardBackwardProps.png)

___

## Day 2

Started the next course in the same specialization. The title of the course is **Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization**.
Today I completed the Week 1 of the course. The course consists of the topics like **Train/dev/test sets**, **Bias and Variance**,
 **Regularization** with techniques like **L2 regularization** and **Dropout regularization**. The rest of the contents were
about dealing with **vanishing/exploding gradients** with **Random Initialization** of weights and **Gradient Check**. 

Some slides snippets from the course are provided below : <br> <br>
 
- **Bias And Variance :** **Bias** refers to the error due to overly simplistic assumptions in the model, causing it to underfit the data.
**Variance** refers to the error due to the model being too sensitive to small fluctuations in the training data, leading to overfitting.
<br> <br> ![biasVariance](Day1_To_10/day2_biasVariance.png) <br> <br>
- **Dropout Regularization :**  **Dropout regularization** randomly deactivates a fraction of neurons during training to prevent overfitting by
forcing the network to learn more robust features. <br> <br> ![dropout](Day1_To_10/day2_dropoutRegularization.png) <br> <br>
- **Gradient Check :** **Gradient check** is a process to verify the correctness of computed gradients by comparing them with numerically approximated gradients.
It is done to ensure that backpropagation is implemented correctly, helping identify bugs in the gradient computation. <br> <br> ![GradientChecking](Day1_To_10/day2_gradCheck.png)

___

## Day 3

Completed the week 2 of the course that I started yesterday. The week was about **Optimization Alogrithms**. This includes the topics like
**Mini-Batch Gradient Descent**, **Gradient Descent With Momentum**, **RMSprop**, **Adam Optimization Algorithm** and **Learning Rate Decay**.
<br> <br>
Putting my understanding on brief with snippets from course slides as follows : <br> <br>

- **Mini-Batch Gradient Descent :** **Mini-batch gradient descent** updates model parameters by computing gradients on small, random subsets (mini-batches)
of the training data, combining the benefits of both **batch** (size = m) and **stochastic** (size = 1) gradient descent. It strikes a balance between fast convergence and stable optimization,
making it well-suited for large datasets and neural networks. <br> <br> ![MiniBatch](Day1_To_10/day3_miniBatch.png) <br> <br>
- **Gradient Descent With Momentum :** It enhances optimization by accumulating a velocity term (**exponentially weighted average** of past gradients) to maintain
direction, reducing oscillations and speeding up convergence, especially in regions with high curvature. <br> <br> ![GradientDescentWithMomentum](Day1_To_10/day3_gradientDescentWithMomentum.png) <br> <Br>
- **RMSprop :** **RMSprop** adapts the learning rate for each parameter by dividing the gradient by the square root of an exponentially weighted average of past squared gradients, 
helping the optimization converge faster and avoid oscillations, especially in non-convex problems. <br> <br> ![RMSProp](Day1_To_10/day3_RMSprop.png) <br> <br>
- **Adam Optimization Algorithm :** It combines the benefits of **Momentum** and **RMSprop** by maintaining both an exponentially weighted average of 
past gradients (like Momentum) and past squared gradients (like RMSprop) to adapt the learning rate for each parameter. This dual mechanism helps Adam achieve faster
convergence with smoother updates, making it well-suited for complex, non-convex problems. <br> <br> ![Adam](Day1_To_10/day3_adamOptimization.png) <br> <br>
- **Learning Rate Decay :** **Learning rate decay** reduces the learning rate progressively during training to ensure faster initial convergence and finer adjustments in later stages, 
preventing the model from overshooting the optimal solution. It improves stability by starting with larger updates and gradually decreasing the step size as the model approaches the minimum.

___

## Day 4

Completed the another course from the specialization : **Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization**.
The remaining portion was about the **Hyperparameter Tuning**, **Batch Normalization**, **Softmax Regression** and **Tensorflow Introduction**. 
<br> <br>
Putting my understanding on brief with snippets from course slides as follows : <br> <br>

- **Hyper-parameter Tuning :** It is the process of optimizing predefined settings like learning rate, batch size, and architecture 
to improve model performance. It involves techniques such as grid search or random search to efficiently explore the hyperparameter space. <br> <br>
- **Batch Normalization :** It is a technique that normalizes activations of each layer within a mini-batch, ensuring stable distributions 
throughout training. It speeds up convergence, reduces internal covariate shift, and helps mitigate issues like vanishing or exploding gradients.
<br> <br> ![batchnorm](Day1_To_10/day4_batchNorm.png)
- **Softmax Regression :** It is a generalization of logistic regression that predicts probabilities across multiple classes by applying the softmax function to convert 
logits into a probability distribution. <br> <br> ![softmax](Day1_To_10/day4_softmax.png) <br> <br>
- **Basic Tensorflow Introduction :** It is an open-source machine learning framework developed by Google that facilitates building, training, and deploying machine l
earning and deep learning models across various platforms. <br> <br> ![Tensorflow](Day1_To_10/day4_tensorflow.png)

___

## Day 5

Today I completed the next course from the specialization : **Structuring Machine Learning Projects**. In this course I learnt to diagnose errors in a machine learning system; 
prioritize strategies for reducing errors; understand complex ML settings, such as **mismatched training/test sets**, and **comparing to and/or surpassing human-level performance**; and apply 
**end-to-end learning**, **transfer learning**, and **multi-task learning**. <br> <br>

Putting simple understanding with snippets from course slides as follows :

- **Mismatched training/test sets :** This occurs when the data distributions of the two sets differ significantly, leading to biased or misleading evaluation
results. This can result in overestimated model performance during training, as the model may not generalize well to the test set due to variations in features,
contexts, or conditions not present in the training data. <br> <br> ![mismatch](Day1_To_10/day5_dataMismatch.png) <br> <br>
- **End-to-end Learning :** It is a machine learning approach where the entire process from input to output is modeled as a single system, allowing the algorithm
to automatically learn all the necessary features and transformations directly from the raw data without the need for manual feature extraction. 
 <br> <br> ![EndToEndLearning](Day1_To_10/day5_endToEndLearning.png) <br> <br>
- **Transfer Learning :** It is a technique where a model pre-trained on one task or dataset is fine-tuned on a different but related task, 
leveraging prior knowledge to improve performance and reduce training time. <br> <br> ![TransferLearning](Day1_To_10/day5_transferLearning.png) <br> <br>
- **Multi-task learning :** It is an approach where a model is trained simultaneously on multiple related tasks, allowing it to leverage shared 
information across tasks to improve generalization and performance. <br> <br> ![multi-task-learning](Day1_To_10/day5_multiTaskLearning.png) 

___

## Day 6

Today I started the next course from the specialization : **Convolutional Neural Networks**. I just had understanding on
the **Edge Detection** and **Padding**. <br> <br>

Putting my understanding along with snippets from the course slides as follows : <br> <br>

- **Edge Detection :** It the process of identifying the boundaries or outlines of objects in an image. It helps in extracting 
structural information, which is crucial for object detection, segmentation, and other computer vision tasks. <br> <br> 
   ![edgeDetectionA](Day1_To_10/day6_edgeDetectionA.png) <br> <br> ![edgeDeteectionB](Day1_To_10/day6_edgeDetectionB.png) <br> <br>
- **Padding :** It refers to adding extra pixels (usually zeros) around the input image to control the spatial dimensions of the output 
feature map. It helps preserve more information at the edges and ensures the output size remains consistent, especially with multiple convolutional layers.
 <br> <br> ![paddingA](Day1_To_10/day6_paddingA.png) <br> <br> ![paddingB](Day1_To_10/day6_paddingB.png) <br> <br>

___

## Day 7

Dived deeper into the **CNN**. Putting understandings on various topics with snippets from course slides as follows : 

- **Strided Convolutions :** It refers to convolution operations where the filter moves across the input with a specified step size (stride),
which can be 1 or greater, determining how much the filter shifts between applications; when the stride is greater than 1, the operation
performs both convolution and downsampling by reducing the spatial dimensions of the output. <br> <br> ![stridedConvolutions](Day1_To_10/day7_stridedConvolution.png) <br> <br>
- **Convolutions on RGB Image :** While we were observing the grayscale images preivously, got to know convolutions on RGB image. <br> <br>
    ![RGBConvolutions](Day1_To_10/day7_convoRGB.png) <br> <br>
- **Types of layer in convolutions :** <br> 1. **Convolution (CONV)** <br> <br> ![ConvolutionLayer](Day1_To_10/day7_ConvNetExampleA.png) <br> <br> 2. **Pooling**
   <br> <br> i. **Max pooling** is a down sampling operation that extracts the maximum value from each region of the input, reducing spatial dimensions while retaining important features.
    <br> <br> ![maxPooling](Day1_To_10/day7_maxPooling.png) <br> <br> ii. **Average Pooling** Average pooling is a downsampling operation that reduces spatial dimensions
    by computing the average value from each region of the input. <br> <br> ![averagePooling](Day1_To_10/day7_avgPooling.png) <br> <br> 3. **Fully Connected** 
    connects every neuron from the previous layer to every neuron in the current layer, learning global patterns across the entire input. <br> <br>
    Representing every layers in one neural network : <br> <br> ![CNN](Day1_To_10/day7_CNNExample.png) <br> <br>

___

## Day 8

Today I implemented **convolutional Neural Networks : Forward Pass** step by step. Attaching the snippets below : <Br> <br>

- ![zeroPad](Day1_To_10/day8_zeroPad.png) <br> <br> ![convSingleStep](Day1_To_10/day8_convSingleStep.png) <br> <br> ![convForward](Day1_To_10/day8_convForward.png) 
    <br> <br> ![poolForward](Day1_To_10/day8_poolForward.png) <br> <br>
 
___

## Day 9

Today I studied about **Classic Networks :** **LeNet-5**, **AlexNet** and **VGG-16**. <br> <br> Putting my understanding with snippets from course slides as follows : <br> <br>

- **LeNet-5 :** It is one of the first neural networks for recognizing images, created to help computers read handwritten numbers. It uses layers that first
find small patterns in images and then combine them to understand bigger details, allowing it to recognize numbers accurately. <br> <br>
    ![LeNet-5](Day1_To_10/day9_LeNet5.png) <br> <br>
- **AlexNet :** It is a deep convolutional neural network developed by **Alex Krizhevsky** in 2012, designed to improve image classification accuracy with a much larger and 
more complex architecture than earlier models like LeNet. It has multiple layers of convolution, pooling, and fully connected layers, using ReLU activation and dropout to 
handle larger datasets, which led to a breakthrough in image recognition on the ImageNet dataset. <br> <br> ![AlexNet](Day1_To_10/day9_AlexNet.png) <br> <br>
- **VGG-16 :** It is a deep convolutional neural network developed by the **Visual Geometry Group (VGG)** that uses a straightforward 
architecture with **16 layers** to achieve high performance in image classification tasks. Its design relies on stacking small 3x3 
convolutional filters and pooling layers, which allows it to capture fine details in images and deliver powerful, accurate image recognition.
<br> <br> ![VGG-16](Day1_To_10/day9_VGG16.png)

___

## Day 10

Today I studied about **ResNet** and **1 X 1 Convolution** and also went through the paper on **AlexNet** : [**ImageNet Classification with Deep Convolutional Neural Networks**](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).
Will discuss the paper after finishing it. For today, let's discuss the following : <br> <br>

- **ResNet :** ResNet is a deep neural network that uses **residual blocks** to make training very deep networks easier and more effective.
Each residual block has a shortcut connection, allowing the input to skip over intermediate layers and be added to the output, which helps
prevent issues like vanishing gradients. This structure lets ResNet learn small improvements (residuals) over the original input, making 
it more efficient at learning complex features without getting "stuck" as the network depth increases. <br> <br> ![residualBlock](Day1_To_10/day10_residualBlock.png) <br> <br>
    ![resNet](Day1_To_10/day10_resNet.png) <br> <br>
- **1 X 1 Convolution :** A **1x1 convolutional layer** applies a filter of size 1 X 1, allowing it to process each pixel individually while combining 
information from different channels. This layer is highly efficient for **reducing or expanding depth** (channel size) and **recomputing feature 
representations** without altering the spatial dimensions, making it useful for computational efficiency and richer feature learning.
    <br> <br> ![1X1Conv](Day1_To_10/day10_1By1Convolution.png) <br> <br>

___

## Day 11

I studied about **Inception Network**, **MobileNet**, **EfficientNet** and got practical advices for using **ConvNets**. <br> <br>

Putting my understanding with some slide snippets from the course as follows : <br> <br>

- **Inception Network :** It is a deep convolutional neural network architecture designed to achieve high accuracy with efficient
computation by combining multiple filter sizes at each layer to capture a range of feature scales. Its core building block,
the **Inception module**, integrates parallel convolution operations of different filter sizes within a single layer, allowing the
network to process features at various scales simultaneously. Together, the Inception modules enable the Inception network to 
improve performance without drastically increasing computational costs. <br> <br> ![InceptionMoudle](Day11_To_20/day11_inceptionModule.png) <br> <br>
    ![inceptionNet](Day11_To_20/day11_inceptionNet.png) <br> <br> 
- **MobileNet :** It is a lightweight convolutional neural network that achieves computational efficiency by using **depthwise separable convolutions**
instead of standard convolutions throughout its layers. In this structure, **the depthwise convolution** applies a single filter per input channel to capture 
spatial information, followed by a **pointwise convolution** to combine these channels, drastically reducing the number of parameters and computations required.
    <br> <br> ![depthwiseSeparableConvolution](Day11_To_20/day11_depthwiseSeparableConvolution.png) <br> <br> ![MobileNet](Day11_To_20/day11_mobileNet.png) <br> <br>
- **EfficientNet :** It is a family of convolutional neural networks that achieves high accuracy and efficiency by uniformly scaling 
network depth, width, and resolution using a compound scaling method. <br> <br> ![efficientNet](Day11_To_20/day11_efficientNet.png) <br> <br>

___

## Day 12

I completed the **Convolutional Neural Networks (CNN)** course from the **Deep Learning Specialization**. The remaining 
contents were about **YOLO**, **U-Net**, **Face Verification & Recognition** and **Neural Style Transfer**.
<br> <br>

Sharing some slide snippets from the course : <br> <br>

 - ![YOLO](Day11_To_20/day12_YOLO.png) <br> <br>
 - ![UNet](Day11_To_20/day12_Unet.png) <br> <br>
 - ![faceRec](Day11_To_20/day12_faceRec.png) <br> <br>
 - ![siameseNet](Day11_To_20/day12_siameseNet.png) <br> <br>
 - ![neuralStyleTransfer](Day11_To_20/da12_neuralStyleTransfer.png) <br> <br>

Additional to these, I also read the research paper : [**You Only Look Once: Unified, Real-Time Object Detection**](https://arxiv.org/abs/1506.02640)

___

## Day 13

Today I started the next course from the **Deep Learning Specialization** and that is **Sequence Models.** Today I just got
introduced with the **Sequence Models**, **Notations** and **Forward Propagation in RNN (Recurrent Neural Networks)**. 
<br> <br>
Putting my understanding and snaps from course notes as follows : <br> <br>

- **Sequence Models :** Sequence models are machine learning models designed to process sequential data, such as time series,
text, or audio, by capturing dependencies between elements across time or position. <br> <br>
   ![sequenceDataExamples](Day11_To_20/day13_sequenceData.png) <br> <br>
- **Recurrent Neural Networks :** Recurrent Neural Networks (RNNs) are a type of neural network made for handling sequences 
of data, like sentences or time series, by remembering information from earlier steps as they process the sequence. They are 
good at tasks like predicting the next word in a sentence but can struggle with remembering things over long sequences. <br> <br>
    ![forwardPropInRNN](Day11_To_20/day13_forPropRNN.png) <br> <br> ![backprop](Day11_To_20/day13_backprop.png) <br> <br>


___

## Day 14

Learnt about **Different Types of RNN**, **Language Modeling with RNN** and **Vanishing Gradients With RNN**. <br> <br>

Putting my understanding and snaps from course notes as follows : <br> <br>

- **RNN Types :** Here are the types of RNNs based on input-output relationships:  

  1. **One-to-One:**  
     - Fixed input and output.  
     - Example: Image classification.  

  2. **One-to-Many:**  
     - Single input with a sequence output.  
     - Example: Image captioning.  

  3. **Many-to-One:**  
     - Sequence input with a single output.  
     - Example: Sentiment analysis.  

  4. **Many-to-Many (Same Length):**  
     - Sequence input and output of the same length.  
     - Example: Video classification frame-by-frame.  

  5. **Many-to-Many (Different Length):**  
     - Sequence input and output of different lengths.  
     - Example: Machine translation.  
  <br> 

  ![RNNTypes](Day11_To_20/day14_RNNTypes.png) <br> <br>


- **Language Modeling With RNN :** Language modeling with RNN involves predicting the next word in a sequence by learning
the conditional probability of words based on the context of preceding words in the text. <br> <br> ![LanguageModel](Day11_To_20/day14_LanguageModeling.png) <br> <br>
- **Vanishing Gradients With RNN :** The vanishing gradient problem in RNNs occurs when gradients diminish during backpropagation
through time, hindering learning in long sequences, and is addressed by using architectures like **LSTMs** or **GRUs** with gating mechanisms to preserve long-term dependencies.
<br> <br> ![Vanishing Gradients](Day11_To_20/day14_vanishingGradientsRNN.png) <br> <br>

___

## Day 15

Read about **GRU** and **LSTM**. Putting my understanding with course slide snaps as follows : <br> <br>

- **GRU** : **A GRU (Gated Recurrent Unit)** is a type of neural network designed to process sequential data by using gates 
to control what information is kept or discarded over time. It helps capture patterns in data like text, speech, or time series by
efficiently managing long-term dependencies. <br> <br> ![GRU](Day11_To_20/day15_GRU.png) <br> <br>
- **LSTM** : **An LSTM (Long Short-Term Memory)** is a type of recurrent neural network that processes sequential data by maintaining
a cell state and using gates to control what information to keep, update, or forget. Unlike a GRU, which combines some gating 
functions, an LSTM has separate forget and input gates along with a cell state, allowing it to potentially handle more 
complex dependencies at the cost of increased computational complexity. <br> <br> ![LSTM](Day11_To_20/day15_LSTM.png) <br> <br>

___

## Day 16

Learnt about **Bidirectional-RNN(BRNN)** and **Deep RNN**. 
<br> <br>
- **Bidirectional RNN :** A Bidirectional Recurrent Neural Network (BRNN) is an RNN architecture that processes data in
both forward and backward directions, allowing the model to capture information from both past and future contexts for each time step.
<br> <br> ![BRNN](Day11_To_20/day16_bidirectionalRNN.png) <br> <br>
- **Deep RNN :** A Deep Recurrent Neural Network (Deep RNN) is a type of RNN that employs multiple stacked layers of recurrent
cells, allowing it to capture more abstract and complex patterns in sequential data compared to a standard, single-layer RNN.
 <br> <br> ![DeepRNN](Day11_To_20/day16_deepRNN.png) <br> <br>

Additional to this, I became familiar with **Tensorflow** and with the help of documentation I trained a s**imple neural network**
and also a **convolutional neural network**. You may visit the notebook : <br> <br>
1. [**Simple Neural Network's Notebook**](Day11_To_20/Neural_Networks_TensorFlow.ipynb) <br> <br>
2. [**Simple CNN Model**](Day11_To_20/CNN_Tensorflow.ipynb)

___

## Day 17

Got Introduced with **Word Embeddings**.


- **Word Embeddings :** **Word embeddings** are numerical vector representations of words that capture their meanings, relationships, 
and contexts in a continuous vector space. They enable NLP models to understand semantic similarities, placing similar words closer together, e.g., *king* and *queen*.
<br> <br> ![wordEmbeddings](Day11_To_20/day17_wordEmbeddings.png) <br> <br>
  **Word embeddings**, a type of **transfer learning**, are pre-trained vector representations of words that capture their meanings and relationships,
enabling efficient handling of textual data in downstream tasks. <br> <br> ![transferLearning](Day11_To_20/day17_transferLearningAndWordEmbeddings.png) <br> <br>
  Word embeddings capture **analogies** as a **property** by encoding relationships such that vector arithmetic reflects semantic connections, 
e.g., king - man + woman ≈ queen. <br> <br> ![analogies](Day11_To_20/day17_analogies.png) <br> <br>

- **Embedding Matrix** is a learnable lookup table where each row corresponds to the vector representation of a word in a vocabulary, used to map words to embeddings in NLP models.

___

## Day 18

Today I went further to the **Word Embeddings** learning about **Word2Vec**, **GloVe** and **Sentiment Classification**.
Highlight of the day was **Sentiment Classification**. So, discussing only it below : <br> <br>

- **Simple Sentiment Classification** is the process of analyzing and categorizing text data to determine the sentiment or 
emotional tone, such as positive, negative, or neutral. <br> <br> ![SentimentClassification](Day11_To_20/day18_sentimentalClassification.png) 
- **Simple Sentiment Classification Model :** <br> <br> ![SimpleModel](Day11_To_20/day18_simpleSentimentalModel.png) <br> <br>
    But this model may predict the sentence like **Completely lacking in good taste ....** kind of sentence as positive sentiment as it takes 
average of the words present. So, to solve it **RNN** is used : <br> <br> ![RNN](Day11_To_20/day18_RNNSentimentalModel.png) <br> <br>
- **Debiasing in Word Embeddings :** It involves modifying vector representations of words to reduce or remove biases related
to gender, ethnicity, or other sensitive attributes while preserving their semantic relationships. <br> <br> ![biases](Day11_To_20/day18_biases.png) <br> <br>
    ![Debiasing](Day11_To_20/day18_debiasing.png)

___

## Day 19

Read about **Sequence to Sequence Models today.**
<br> <br>
- **Sequence to Sequence Models :** It transform an input sequence into an output sequence using an encoder to encode the input and a decoder 
to generate the output, making them suitable for tasks like translation and summarization.
 <br> <br> ![Seq2SeqModel](Day11_To_20/day19_seq2seqModel.webp)
  <br> <br>

- **Beam Search :** **Beam search** is a heuristic search algorithm that explores multiple possible sequences simultaneously, keeping only the top
_B_ (beam width) most promising candidates at each step, making it efficient for tasks like machine translation and text generation. <br> <br>
    ![BeamSearch](Day11_To_20/day19_beamSearch.png) <br> <br>

___

## Day 20

Read about **Error Analysis On Beam Search** and **Attention Model**
<br>  
- **Error Analysis On Beam Search** : <br> <br> ![ErrorAnalysisOnBeamSearch](Day11_To_20/day20_errorAnalysisBeamSearch.png) <br> <br>
- **Attention Model** : It is a mechanism in deep learning that enables a network to focus on specific parts of the input when making predictions, 
mimicking human attention. It assigns varying weights to different input elements, enhancing the model's ability to handle tasks like language 
translation, image captioning, and sequence-to-sequence problems effectively. <br> <br> 
    ![attentionModel](Day11_To_20/day20_attentionModel.png) <br> <br> 
- **Attention Model For Speech Recognition :** <br> <br> ![SpeechRecognition](Day11_To_20/day20_speechRecognition.webp) <br> <br>
- **Trigger Word Detection :** <br> <br> ![TriggerWord](Day11_To_20/day20_triggerWordDetection.png) 

___

## Day 21

Got the Transformer Intuition through **Self-Attention** and **Multi-head Attention**.
<br> <Br>
- **Self-Attention :** **Self-attention** in transformers allows each word (or token) in a sequence to focus on other words in the same
sequence to capture contextual relationships, regardless of their distance. It computes a weighted representation of the sequence by 
assigning importance scores (attention weights) based on the relevance between tokens, enabling the model to understand the meaning of a word in context. <br> <br>
 ![SelfAttention](Day21_To_30/day21_selfAttention.png) <br> <br>
- **Multi-Head Attention :** **Multi-head attention** is an extension of self-attention that uses multiple parallel attention mechanisms (heads) 
to capture different aspects of relationships between tokens in a sequence. Each head independently computes self-attention, and their outputs are 
concatenated and transformed to enhance the model's ability to learn complex, diverse patterns in the data. <br> <br> ![MultiHeadAttention](Day21_To_30/day21_multiHeadAttention.png) 

___

## Day 22
Completed the **Deep Learning Specialization**. The rest of the portion was about **Transformer**, 
<br> <br>
- **Transformer :** **Transformers** are a deep learning architecture designed to process sequential data by leveraging the self-attention mechanism, 
which enables the model to weigh the importance of different input tokens relative to one another, regardless of their position. 
This architecture has revolutionized natural language processing (NLP) and beyond, powering models like GPT and BERT due to its efficiency in parallel 
processing and ability to capture long-range dependencies. <br> <br> ![Tranformer](Day21_To_30/day22_Transformer.png) <br> <br>

___

## Day 23

Started working on a project [**Brain Tumor Detection System**](https://github.com/iamshishirbhattarai/Machine-Learning/tree/main/Brain%20Tumor%20Detection%20System). 
Today I just loaded the dataset, visualized and performed data augmentation. <br> <br>
- **Code Snippet :** <br> <br> ![codesnippet](Day21_To_30/day23_codeSnippet.png) <br> <br>  Sorry for using // instead of # for comments. <br> <br>
- **Categories Visualization :** <br> <br> ![categores](Day21_To_30/day23_categories.png) <br> <br>

You may visit the ongoing notebook through the link above.

___

## Day 24

Well I trained the model and obtained the **train accuracy** of **97.5%** and **test accuracy** of **94.1%**. It's not much bad, but the
system obtained only **80.9%** which might be low. **Lower recall** in a brain tumor detection system means more tumors may go undetected,
potentially delaying treatment and leading to severe or fatal outcomes. But let's see what I can do in further days to improve this.

- **Snippet :** <br> <br> ![ModelSnippet](Day21_To_30/day24_modelSnippet.png) <br> <br>

 Visit the full project at : [**Brain Tumor Detection System**](https://github.com/iamshishirbhattarai/Machine-Learning/tree/main/Brain%20Tumor%20Detection%20System).
 
___

## Day 25

Tried implementing **Transfer Learning**. For this, I took a pretrained **MobileNetV2** on **ImageNet**  to classify **CIFAR-10** images.

- ![TransferLearning](Day21_To_30/day25_transferLearningCode.png) <br> <br>
 

___

## Day 26

Started a new project called [**Pneumonia Prediction Using CNN**](https://github.com/iamshishirbhattarai/Pneumonia-Prediction-Using-CNN). Today just simply loaded the dataset, preprocessed
and roughly created the model. Have to make improvements, which I will see in coming days. <br> <br>

- **Normal Vs Pneumonia Image :** <br> <br>![NormalVsPneumonia](Day21_To_30/day26_Normal_Vs_Pneumonia.png) <br> <br>
- **Model Definition :** <br> <br> ![Model](Day21_To_30/day26_pneumoniaModel.png)

___

## Day 27

Tried improving the project that I started yesterday. But, couldn't do though I tried it the whole day. There might be some
issues or lacking, which I might figure out later. For now, ending up the project. <br> <br>
- ![PneumoniaMOdelRevised](Day21_To_30/day27_PnemoniaModelRevised.png) <br> <br>

Visit the Porject : [**Pneumonia Prediction Using CNN**](https://github.com/iamshishirbhattarai/Pneumonia-Prediction-Using-CNN)


___

## Day 28

Today had some fun with **YOLO**. Used **YOLOv8** from **ultralytics**. During this also revisited the 
[**You Only Look Once: Unified, Real-Time Object Detection**](https://arxiv.org/abs/1506.02640) paper.

- ![code](Day21_To_30/day28_yoloBeginner.png) <br> <br>
- **Output :** <br> <br> ![objectDetection](Day21_To_30/day28_objectDetection.png) <br> <br>

You may visit the notebook : [**YOLO Beginner Notebook**](Day21_To_30/day28_YOLO_Beginner.ipynb)

___

## Day 29 

Tried to train the **YOLOv8** with custom dataset for the whole day but got problems which I will figure out tomorrow.
Other than that, started **NLP Specialization** from **Coursera**. Today, I learnt about **Vocabulary**, **Feature Extraction**,
**Negative And Positive Frequencies** and **Preprocessing**. 

- **Vocabulary and Feature Extraction :** <br> <br> ![Vocabulary](Day21_To_30/day29_Vocabulary.png) <br> <br> ![FeatureExtraction](Day21_To_30/day29_featureExtraction.png) <br> <br>
- **Positive and Negative Frequencies :** <br> <br> Since in **Sparse Representation** model has to learn large number of parameters 
resulting **large training time** and **large prediction time**, the **Negative and Positive Frequencies** comes to rescue. How? Might be clarified
with the snapshot from the course below : <br> <br> ![PosNegFreq](Day21_To_30/day29_posNegFreq.png) <br> <br>
- **Pre-Processing :** Involves techniques like eliminating **stop words, punctuation, handles, URLs, Stemming and Lowercasing.** <br> <br>
      1. **Stop words**: Common words (e.g., "the," "and") removed in text processing to focus on meaningful terms.  
      2. **Punctuation**: Symbols (e.g., ".", ",") often removed during preprocessing to simplify text analysis.  
      3. **Handles**: Usernames in social media (e.g., "@user") excluded to reduce noise.  
      4. **URLs**: Web links stripped out for clean text data.  
      5. **Stemming**: Reduces words to their base form (e.g., "running" → "run"). <br> 
      6. **Lowercasing**: Converts all text to lowercase to ensure uniformity.

___

## Day 30

Completed the first week from the first course of **NLP Specialization**. THe rest of the stuff was about **Logistic Regression**
to classify whether the given tweet is positive or negative. <br> <br>

- ![LR](Day21_To_30/day30_LR.png) <br> <br> ![LRTraining](Day21_To_30/day30_LRTraining.png) <br> <br>

ALso tried fine tuning the **yolov8** with [**Road Detection**](https://www.kaggle.com/datasets/princekhunt19/road-detection-imgs-and-labels/data) dataset. 
Though, it's not much better, but it at least was error free today. Will improve it more tomorrow. Some output from the model is as follows : <br> <br> 
- ![Output1](Day21_To_30/day30_output1.png) <br> ![Output2](Day21_To_30/day30_output2.png) <br> ![Output3](Day21_To_30/day30_output3.png) 

___

## Day 31

I prepared the final notebook for fine tuning of **Yolov8** which you may visit here : [**Fine Tuning yolov8**](https://github.com/iamshishirbhattarai/Fine-Tuning-YOLOV8) <br> <br>

On the other side, today I learnt about **Conditional Probability**, **Baye's Rule** and **Naive Bayes**. <br> <br>

- **Conditional Probability :** Conditional probability is the likelihood of an event occurring given that another event has already occurred. <br> <br> 
  ![ConditionalProbability](Day31_To_40/day31_conditionalProbability.png) <br> <br>
- **Baye's Rule :** It describes how to update the probability of an event based on new evidence, incorporating prior knowledge and the likelihood of the evidence. <br> <br>
   ![BayesRule](Day31_To_40/day31_bayesRule.png) <br> <br>
- **Naive Bayes :** Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem, assuming feature independence, commonly used for classification tasks like spam detection and sentiment analysis.
   <br> <br> ![NaiveBayes](Day31_To_40/day31_naiveBayes.png) <br> <br>
___

## Day 32

Learnt about **Laplacian Smoothing**, **Log Likelihood**, **Training and Testing Naive Bayes**, **Applications** and **Error Analysis**
<br> <br>

- **Laplacian Smoothing :** Laplacian smoothing is a technique used in probability and statistics to handle zero probabilities by adding a 
small constant (usually 1) to each count, ensuring that no event has a zero probability. <br> <br> ![LaplacianSmoothing](Day31_To_40/day32_laplaceSmoothing.png) <br> <br>
- **Log Likelihood :** It is the logarithm of the likelihood function which is done to simplify computations, avoid numerical underflow when multiplying many small
probabilities, and to make optimization easier when fitting the model to data. <br> <br> ![LogLikelihood](Day31_To_40/day32_logLikelihood.png) <br> <br>
- **Training Naive Bayes :** <br> <br> ![TrainingNaive](Day31_To_40/day32_trainingNaiveBayes.png) <br> <br>
- **Applications of Naive Bayes :** <br> <br> 1. Sentiment Analysis <br> 2. Author Identification <br> 3. Information Retrieval <br> 4. Word disambiguation, etc.
<br> <br>

___

## Day 33

Learnt about **Vector Space**, **Euclidean Distance** and **Cosine Similarity** from **NLP Specialization**. <br> <br>

- **Vector Space :** 
A **vector space** is a way to represent data mathematically, where words or documents are treated as points or vectors in 
a multi-dimensional space, making it easier to analyze their relationships. In a **word-by-word approach**, a co-occurrence
matrix is created, where each word is compared to every other word to capture their contextual relationships. In contrast,
a **word-by-docs** matrix links words to documents, showing how frequently a word appears in each document to determine its relevance. 
Together, these methods help in tasks like understanding word associations or analyzing document similarity.
<br> <br>
- **Euclidean Distance :** It is the straight-line distance between two points in a multi-dimensional space, calculated
as the square root of the sum of squared differences between corresponding coordinates. <br> <br> ![EuclideanDistance](Day31_To_40/day33_EuclideanDistance.png)<br><br>
- **Cosine Similarity :** It measures the angle between two vectors in a multi-dimensional space, indicating how similar they are based on their 
direction, regardless of their magnitude. <br> <br> ![CosineSimilarity](Day31_To_40/day33_CosineSimilarity.png)

___

## Day 34

Learnt about **Manipulating Vector Space** and **PCA Algorithm.**

- **Manipulating Vector Space**: Below demonstrates an example of how vector space can be used to find capital of the city
   when one of them is provided. <br> <br> ![ManipulatingVector](Day31_To_40/day34_manipulatingVector.png) <br> <br>
- **PCA Algorithm :** It is a dimensionality reduction technique that identifies the principal components
of a dataset by calculating the eigenvectors and eigenvalues of its covariance matrix. The eigenvectors represent the 
directions of maximum variance (principal components), while the **eigenvalues** quantify the amount of variance captured by
each corresponding **eigenvector**. <br> <br> ![PCA1](Day31_To_40/day34_PCA1.png) <br> ![PCA2](Day31_To_40/day32_PCA2.png) <br> <br>

___

## Day 35

Completed the first course of **NLP Specialization** : **S**. 
The remaining portion was about **K-nearest neighbors**, **Hash Tables and Functions**, **Locality Sensitive Hashing**, etc.

- **K-nearest Neighbors :** It is a machine learning algorithm that classifies a data point based on the majority class of its K
closest points in the feature space.  <br> <Br>
- **Hash Table :** <br> <br> ![hashValue](Day31_To_40/day35_hashValue.png) <Br> <br>
- **Locality Sensitive Hashing :** It is a technique used to perform approximate nearest neighbor searches by hashing similar 
data points into the same bucket with high probability, reducing the search space. <br> <br> ![LSH](Day31_To_40/day35_hashValue.png) <br> <Br>
**Multiple planes** are used in locality-sensitive hashing (LSH) to partition the data into distinct subspaces. Each plane is essentially a 
hyperplane that divides the data into two regions, which helps in efficiently hashing and indexing high-dimensional vectors.  <br> <Br> ![multiplePlanes](Day31_To_40/day35_multiplePlanes.png) <br> <br>
- **Approximate Nearest Neighbors :** It is a technique used to quickly find vectors that are close to a given query vector
in high-dimensional spaces, by using approximate methods like locality-sensitive hashing (LSH) to reduce computation time 
and memory usage, while sacrificing some accuracy.

___

## Day 36

Started the new course withing **NLP Specialization** : **Natural Language Processing with Probabilistic Models**. 
Today Just learnt about **AutoCorrect** and **Building The Model**.

- **Autocorrect :** It automatically adjusts misspelled or mistyped words to their correct form while typing;
for example, typing "definately" might be corrected to "definitely." <br> <br> ![autoCorrect](Day31_To_40/day36_autoCorrect.png) <br> <br>
- **Building The Model :** Autocorrect involves the following process to build the model: <br> <br> ![BuildingModel](Day31_To_40/day36_buildingModel.png) <br> <Br>

___

## Day 37

Learnt about **Minimum Edit Distance Algorithm.** <br> <br>
- The **Minimum Edit Distance Algorithm** (or Levenshtein Distance) calculates the smallest number of operations
(insertions, deletions, and replacements) required to convert one string into another. It uses dynamic programming to build a matrix,
where each cell represents the edit distance for substrings, enabling an efficient bottom-up computation of the result.
 <br> <br> ![MinimulEditDistance](Day31_To_40/day37_MED1.png) <br> <br> ![Minimum Edit Distance](Day31_To_40/day37_MED2.png)

___

## Day 38

Learnt about **POS tagging** and **Markov Chains**.

- **POS Tagging :** **Part Of Speech Tagging** is the process of assigning a part of speech (e.g., noun, verb, adjective) 
to each word in a sentence based on its context and definition. It is a fundamental task in natural language processing (NLP)
that helps in syntactic and semantic analysis. Machine learning algorithms and probabilistic models, like Hidden Markov Models (HMMs), are often used for efficient tagging.
<br> <br> ![POS](Day31_To_40/day38_POS.png) <br> <br>
- **Markov Chain :** It is a mathematical model that describes a system transitioning between states based on certain probabilities. 
The key property is that the probability of moving to the next state depends only on the current state (memoryless). Transition probabilities determine how likely the system moves from one state to another, and the initial state probability 
(**π** ) specifies the probability of starting in a specific state. <br> <br> ![Transition](Day31_To_40/day38_TransitionMatrix.png) <br> <br>


___

## Day 39

Learnt the following: <br> <br>

- **Hidden Markov Model :** A **Hidden Markov Model (HMM)** is a statistical model used to represent systems that have hidden states 
influencing observable outputs, where the transitions between hidden states follow a Markov process. It is characterized by transition
probabilities between states, emission probabilities linking hidden states to observations, and initial state probabilities. <br> <br>
- **Transition Probabilities :** **Transition probabilities** in a Hidden Markov Model (HMM) represent the likelihood of transitioning 
from one hidden state to another in a sequence. <br> <br> ![TransitionProbabilities](Day31_To_40/day39_TransitionProbabilities.png) <br> <br> 
  **Populating Transition Probabilities** involves estimating the likelihood of transitioning from one hidden state to another, often using frequency 
counts from training data or predefined probabilities. <br> <br> ![PopulatingTP](Day31_To_40/day39_populatingTransitionProbabilities.png) <br> <br>
  **Smoothing** in the context of populating transition probabilities ensures that even rare or unseen transitions are assigned a small probability to 
avoid zero-probability issues during model computations. <br> <br> ![smoothing](Day31_To_40/day39_smoothingTransitionProbabilities.png) <br> <br>
- **Emission Probabilities :** **Emission probabilities** in an HMM represent the likelihood of an observed symbol being generated from a specific
hidden state, differing from transition probabilities, which define the likelihood of moving from one hidden state to another. <br> <br> ![EmissionProbabilities](Day31_To_40/day39_emmissionProbabilities.png) <br> <br>
    **Populating emission probabilities** involves estimating the likelihood of each observed symbol being produced by a given hidden state, often
using frequency counts from training data and applying smoothing techniques to handle unseen observations. <br> <br> ![PopulatingEP](Day31_To_40/day39_populatingEmmissionProbabilities.png) <br> <br>

---

## Day 40

Learnt about **Viterbi Algorithm**.

The **Viterbi algorithm** is a dynamic programming algorithm used to find the most probable sequence of hidden states in a Hidden Markov Model (HMM), given a sequence of observed events. It is widely applied in areas such as speech recognition, bioinformatics, and natural language processing.

- ![Viterbi](Day31_To_40/day40_viterbiAlgo.png)
### Steps of the Viterbi Algorithm

- ### Initialization

    The algorithm begins by calculating the initial probabilities for each state based on the starting probabilities of the HMM and the likelihood of observing the first data point. These probabilities are stored as the starting values for the most likely path calculation.

- ### Forward Pass

    For each subsequent observation in the sequence, the algorithm computes the probability of being in each state by considering: <br> <br>

  - The probabilities of all possible previous states.
  - The transition probabilities between states.
  - The emission probabilities of observing the data point from each state.
    The most likely previous state for each current state is recorded to help reconstruct the optimal path later.

- ### Backward Pass

    After processing all observations, the algorithm traces back through the recorded states to determine the most probable sequence of hidden states. This step reconstructs the sequence that maximizes the probability of the observed data.

- ### Example Use Cases

  - Speech recognition: Identifying words from audio signals.
  - Bioinformatics: Predicting gene sequences.
  - Natural language processing: Part-of-speech tagging.

The Viterbi algorithm efficiently solves problems involving HMMs by combining probabilistic reasoning and dynamic programming.

___

## Day 41

Learnt about **NGram** and **Probabilities**.

- **N-Gram** : An **n-gram** is a contiguous sequence of n items(words, characters, or tokens) from a given text or speech.
  <br> <br> ![ngram](Day41_To_50/day41_NGram.png) <br> <br>
- **Probabilities :** <br> <br> ![unigram](Day41_To_50/day41_uniGramProb.png) <br> ![bigram](Day41_To_50/day41_biGramProb.png) <Br> 
  ![trigram](Day41_To_50/day41_triGramProb.png) <br> ![ngramprob](Day41_To_50/day41_nGramProb.png) <br> <br>

___

## Day 42

Learnt about **Sequence Probabilities.**

- Sequence probabilities measure the likelihood of a sequence of events (e.g., words in a sentence) occurring in a specific order,
calculated as the product of conditional probabilities of each event given its predecessors. However, exact computation becomes infeasible
for long sequences due to the exponential growth of dependencies and the need for vast amounts of data to estimate probabilities accurately.
To address this, **n-gram models** approximate sequence probabilities by considering only a fixed number of preceding events (n-1), 
significantly reducing computational complexity and data requirements. While this approach simplifies computation, it assumes independence
beyond the n-gram context, potentially missing longer-range dependencies. <br> <Br> Despite its limitations, n-gram models are foundational in natural 
language processing and paved the way for more advanced methods like neural language models. <br> <br> ![seqProb](Day41_To_50/day42_seqProbabilities.png) <br>
 ![seqProbShortComings](Day41_To_50/day42_seqProbShortComings.png) <br> ![SeqProbApprox](Day41_To_50/day42_seqProbApprox.png)

___

## Day 43

Learnt about **N-gram Models**, **Model Evaluation**, **Out Of Vocabulary** and **Smoothing**.
<br> <br>
- **N-gram Model :** The N-gram model predicts the probability of a word sequence based on the conditional probability of each word given its
    n−1 preceding words. <br> <br>

  **Steps and Explanations**

    1. **Count Matrix** <br>
    Records the frequency of n
n-grams in the dataset to capture patterns.
   2. **Probability Matrix** <br>
   Converts n-gram frequencies into conditional probabilities by normalization.
   3. **Language Model** <br> 
   Uses the probability matrix to predict or evaluate the likelihood of sentences.
   4. **Log Probability**  <br> Avoid Underflow by
   Applying logarithms to probabilities to ensure numerical stability in computations.
   5. **Generative Language Model** <br> 
   Generates new text by sampling from learned 
   n-gram probabilities. <br> <br> ![GenerativeLM](Day41_To_50/day43_generativeLanguageModel.png) <br> <br>

- **Perplexity (an Evaluation Metric) :** It is a measurement of how well a probabilistic model predicts a dataset, 
often used in language modeling, where lower values indicate better predictive performance. <br> <br> ![perplexity](Day41_To_50/day43_bigramPerplexity.png)
  <br> ![logPerplexity](Day41_To_50/day43_logPerplexity.png) <br> <br>
- **Out Of Vocabulary :** **Out of vocabulary (OOV)** refers to words or tokens in a dataset that are not present in a model's training vocabulary, making them unrecognizable to the model.
    One of the way to handle is by using **< UNK >**. <br> <br> ![UNK](Day41_To_50/day43_usingUNK.png) <br> <br>
- **Smoothing :** It is a technique used in language models to assign non-zero probabilities to unseen events, preventing them from being entirely excluded. <br> <br>
    ![Smoothing](Day41_To_50/day43_smoothing.png) <br> <br>
- **Backoff :** It is a strategy where the model defaults to simpler, lower-order models when higher-order n-grams are not found in the training data. <br> <br>
    ![backoff](Day41_To_50/day43_backoff.png) <br> <Br>

___

## Day 44

Revised about **Word Embedding**. <br> <br>

-  **Word Embedding :** **Word embeddings** represent words as dense vectors in a continuous space, capturing their meaning and 
relationships based on usage in context. This makes them far more efficient and meaningful than **integer** or **one-hot encoding**, 
which are sparse and lack the ability to convey semantic or contextual information. <br> <br>
- **Continuous bag-of-words :** **Continuous Bag of Words (CBOW)** predicts a target word based on the surrounding context words, using
a sliding window approach to learn word representations. <br> <br> ![corpusToTraining](Day41_To_50/day44_corpusToTraining.png) <br> <br>

___

## Day 45

Diving deeper into **CBOW**.

- **Cleaning and Tokenization :** It involve preparing text by removing unwanted elements 
(like punctuation, emojis, or stopwords) and splitting it into meaningful units such as words or tokens for analysis.
 <br> <br> ![cleaningAndTokenization](Day41_To_50/day45_cleaning&Tokenization.png) <br> <br>
- **Sliding Window :** The **sliding window** technique defines a fixed-size context window that moves across the text, using the words within the window (**context**) to predict the target word at the **center**.
    <br> <BR> ![SLIDINGWINDOW](Day41_To_50/day45_slidingWindow.png) <br> <br>
- **Transforming Words Into Vectors :** <br> <br> ![CenterWords](Day41_To_50/day45_centerWordsIntoVectors.png) <br> ![ContextWords](Day41_To_50/day45_contextWordIntoVectors.png) <br> <br>
- **Architecture :** <br> <br> ![CBOWArchitecture](Day41_To_50/day45_CBOWArchitecture.png) <br> <br>

___

## Day 46

Continued **CBOW**. <br> <br>

- **Dimensions on CBOW** <br> <br> ![dimensionSingle](Day41_To_50/day46_dimensionSingle.png) <br> ![DimensionBatch](Day41_To_50/day46_dimensionBatch.png) <br> <br>
- **Training Process** <br> <br> 1. Forward Propagation <br> 2. Cost <br> 3. Backpropagation and gradient descent.
  <br> <br>
- **Evaluation** <br> <br>  **Intrinsic Evaluation**: CBOW can be intrinsically evaluated by measuring word similarity or analogy tasks, assessing how well the model captures semantic and syntactic relationships between words.  
   <br> ![intrinsicEvaluation](Day41_To_50/day46_intrinsicEvaluation.png) <br> 
  <br>  **Extrinsic Evaluation**: CBOW can be extrinsically evaluated by incorporating its embeddings into downstream tasks like text classification or sentiment analysis and measuring task-specific performance metrics.
    <br> <br> ![ExtrinsicEvaluation](Day41_To_50/day46_extrinsicEvaluation.png) <br> <Br>

___

## Day 47

Today just did lab portion of the **CBOW** and completed the second course from **NLP Specialization**. <br> <br>

- **Backpropagation in CBOW** <br> <br> ![BackProp](Day41_To_50/day47_backPropTrain.png) <br> <br>

___

## Day 48

Started another course on **NLP Specialization : Natural Language Processing with Sequence Models**. Learnt about **Embedding Layer**
, **Mean Layer** and **N-Gram VS RNN**. <br> <Br>

- **Embedding Layer and Mean Layer :** An **embedding layer** takes items like words or IDs and turns them into meaningful numbers 
(vectors) that the computer can understand, making it easier to process things like text or categories. On the other hand, a 
**mean layer** takes a group of these numbers (vectors) and calculates their average to summarize the information. In simple terms, 
the embedding layer creates representations, while the mean layer combines them into a single summary.
<br> <br> ![embeddingLayer](Day41_To_50/day48_embeddingLayer.png) <br> ![MeanLayer](Day41_To_50/day48_meanLayer.png) <br> <br>
- **N-Gram VS RNN :** **N-grams** analyze fixed-size chunks of text, capturing local patterns but failing to consider long-term dependencies
or context beyond their window size, which limits their ability to handle complex language structures. **RNN**s, on the other hand, process 
sequences one step at a time, maintaining a hidden state that carries information from previous inputs, enabling them to capture long-range
dependencies and context. This ability to model sequential data dynamically makes RNNs a better choice for tasks like language modeling and 
machine translation compared to the rigid structure of n-grams. <br> <br>
  ![ngram](Day41_To_50/day48_ngram.png) <br> ![RNN](Day41_To_50/day48_RNN.png) <br> <Br>
___

## Day 49

Revised about **Vanilla RNN**, **GRU**, **Bi-directional RNN** and **Deep-RNN**.
<br> <br> 
- Recurrent Neural Networks (RNNs) are designed to process sequential data by passing information from one step to the next. A **Vanilla RNN**
is the basic version, but it struggles with learning long-term dependencies due to issues like vanishing gradients.  <br> <br> ![VanillaRNN](Day41_To_50/day49_VanillaRNN.png) <br> <br>
- **Gated Recurrent Units (GRUs)** improve this by using gates to control which information is remembered or forgotten, making them more efficient for capturing longer patterns. 
<br> <Br> ![GRU](Day41_To_50/day49_GRU.png) <br> <br>
- **Bidirectional RNNs** process sequences both forward and backward, which helps when context from both past and future is needed, such as in language translation. 
<br> <br> ![BRNN](Day41_To_50/day49_BRNN.png) <br> <br>
- **Deep RNNs** stack multiple RNN layers, making them capable of learning more complex patterns in the data. <br> <br> ![DeepRNN](Day41_To_50/day49_DeepRNN.png) <br>
___

## Day 50

With the revision of **LSTM**, learnt about **Training and Evaluating NER (Named Entity Recognition**. <br> <br>

- ![TrainingNER](Day41_To_50/day50_trainingNER.png) <br> ![EvaluatingNER](Day41_To_50/day50_evaluatingNER.png) <br> <Br>

___

## Day 51

Learnt about **Siamese Network**. <br> <br>
- A **Siamese network** is a neural network architecture designed to compare two inputs by processing them in parallel with shared weights. 
  It generates embeddings for both inputs, which are then compared using a similarity metric to determine their relationship.  
 <br>  For example, in a question similarity task, two questions are passed through an embedding layer and an LSTM to extract their 
 vector representations. These representations are compared using cosine similarity, producing a similarity score between \(-1\) and \(1\). A threshold (\(tau)) 
is used to classify the questions as similar or different, depending on whether the score is above or below the threshold. 
This approach is effective in tasks like detecting duplicate questions or measuring semantic similarity between texts. <br> <br> ![SiameseNetwork](Day51_To_60/day51_siameseNetwork.png) <br>

____

## Day 52

Learnt about **Triplet Loss** and **One-Shot Learning**. <br> <br>

- **Triplet Loss :** **Triplet Loss** is a loss function used in metric learning to ensure that similar items are closer in the embedding 
space while dissimilar items are farther apart, by comparing an anchor, a positive sample, and a negative sample. It minimizes the cosine 
similarity between the anchor and positive sample while maximizing the cosine similarity between the anchor and negative sample, enforcing a margin for better separation. 
**Hard Negative Mining** is a technique applied to Triplet Loss, focusing on the most difficult negative samples, such as the **Closest Negative** (most similar negative) 
or the **Mean Negative** (average similarity of negatives), which helps the model learn more discriminative features and improves its performance on tasks like image retrieval and face verification.
 <br> <br> ![tripletLoss](Day51_To_60/day52_tripletLoss.png) <br> ![hNM](Day51_To_60/day52_hardNegativeMining.png) <br> <br>
- **One Shot Learning :** It is a machine learning paradigm where the model is trained to recognize patterns or classify data after being shown only a single
example per class. It typically leverages techniques like Triplet Loss or Siamese networks to generalize from limited data by comparing new inputs to the known 
example and determining their similarity. <br> <br> ![oneshot](Day51_To_60/day52_oneShot.png) <br> <br> 

___

## Day 53

Completed the **Natural Language Processing with Sequence Models** course. Today just did programming assignment on **Siamese Network.**
<br> <br>
- ![SiameseNetworkProgram](Day51_To_60/day53_siameseNetworkProgram.png) <br>
___

## Day 54

Explored Shortcomings of **Seq2Seq Models** and read [**Neural Machine Translation
By Jointly Learning To Align And Translate**](https://arxiv.org/pdf/1409.0473) paper. <br> <br>

- **Shortcomings of Seq2Seq :** Seq2seq models face a major limitation due to their reliance on a fixed-length context vector, which struggles to retain detailed information from long input sequences. Additionally, as the sequence length increases, the model's performance decreases, making it less effective for handling complex or lengthy data.
  <br> <br> ![ShortcomingsSeq2Seq](Day51_To_60/day54_informationBottleneck.png) <br> <br>
- **Solution :** <br> <br> ![focusAttention](Day51_To_60/day54_focusAttention.png) <br>
  <br> The paper "**Neural Machine Translation by Jointly Learning to Align and Translate**" introduces the attention 
 mechanism to address the fixed-length context vector limitation in seq2seq models. As shown in the **first image**, attention enables the
 decoder to use a weighted sum of all encoder hidden states (instead of just one context vector) by 
  dynamically assigning importance weights based on the decoder's previous state.  <br> <bR>The **second image** details how these weights, computed using a feedforward network followed by a softmax function, allow the model to focus on the most relevant parts of the input sequence for each decoding step. This innovation improves translation performance, especially for longer sequences, by enabling a more context-aware and flexible approach.
  <br> <br> ![howToUseAllHiddenStates](Day51_To_60/day54_howToUseAllHiddenStates.png) <br> 
   ![attentionInDepth](Day51_To_60/day54_attentionInDepth.png) <br> <br> **Performance Comparision** <br> <br>![Performance](Day51_To_60/day54_performance.png)

___

## Day 55

Started the paper [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762). Learnt about few concepts as an individual
topic. Today' understanding includes **Input Embedding** , **Positional Encoding**, **Scaled-dot product attention** and **Self-attention**.
<br> <br> 
- **Input Embedding** converts each word or token in the input sequence into a dense vector representation that captures its semantic meaning. 
  <br> <br> **Positional Encoding** adds information about the position of each token in the sequence to its embedding, ensuring the model can 
  understand word order since transformers lack inherent sequential awareness. <br> <br>  ![positional Encoding](Day51_To_60/day55_positionalEncoding.png)_Source : https://www.youtube.com/watch?v=bCz4OMemCcA_
  <br> <br>
- **Scaled-dot product attention :** It finds the importance of each word by comparing queries and keys, scales the result 
 for stability, normalizes it, and uses it to combine the values for the final output. <br> <br> ![scaledDotProduct](Day51_To_60/day55_scaledDotProduct.png) <br> <br> 
 **Query :** What the current word is looking for in the context.
 <br> <br> **Key:** Describes the content of each word in the context.
<br> <br> **Value:** The actual information carried by each word. <br> <br>
- **Self-attention :** It allows each word in a sequence to focus on and weigh the importance of all other words to understand context and relationships.
 It uses **scaled dot-product attention** to calculate how much focus each word (query) should give to all other words (keys) in the sequence, combining their information
 (values) based on the resulting attention scores.

___

## Day 56

Explored other aspects of **Transformers.** <br> <br>

- **Multi-Head Attention :** It splits the input into multiple **heads** each independently focusing on different aspects of the data.
  This allows the model to capture different patterns and relationships by focusing on various aspects of inputs in parallel.
  <br> <br> ![multiHeadAttention](Day51_To_60/day56_MultiHeadAttention.png) <br> <br>
- **Masked-Head Attention :** Ensures that the model can only attend to previous tokens when generating a sequence, preventing it from 
  cheating by looking at future words. <br> <br> ![MaskedHeadAttention](Day51_To_60/day56_maskedHeadAttention.png) <br> <Br>
- **Training and inference :** In training, the Transformer processes the entire sequence at once, using self-attention to compute relationships between all tokens in parallel, 
enabling efficient learning of global dependencies. Positional encodings are added to the input embeddings to preserve order information. During inference, the model
generates tokens one at a time, using a masked self-attention mechanism to ensure it only considers previously generated tokens for each new prediction. This approach ensures 
the model maintains causal consistency while generating sequences. <br> <br> ![Transformer](Day51_To_60/day56_Transformer.png) <br> <br>
  **Norm** here is the **Layer Normalization** which calculates the mean and variance of activations within individual example and normalizes the
 activations of that specific example based on its own internal statistics.

___

## Day 57

Started Coding the transformer from scratch. Today almost implemented Encoder part. 

- ![InputPositional](Day51_To_60/day57_inputPositionalCode.png) <br> ![LayerFF](Day51_To_60/day57_layerFeedForwardCode.png) <br>
 ![MultiHeadCode](Day51_To_60/day57_multiHeadCode.png) <br> ![ResiudalEncoder](Day51_To_60/day57_residualEncoderCode.png)
 <br> 
___

## Day 58


Completed the model building of the transformer.

- ![decoder](Day51_To_60/day58_decoderProjectionLayer.png) <br> ![transformerClass](Day51_To_60/day58_transformerClass.png) <br>
  ![BuildTransformer](Day51_To_60/day58_buildTransformer.png) <br>

___

## Day 59

Coded the dataset portion. <br> <br>
- ![DataSet](Day51_To_60/day59_dataSetCode.png)

___

## Day 60

Got problem with the Scratch Implementation, will get back to this soon. <br> <br>
Today, explored how to perform an **OCR** using **pytesseract**. <br> <br>

- ![OCR](Day51_To_60/day60_OCR.png) <br> <br>

___

## Day 61
Will be working on the scratch implementation on the background and talk about it when finished. Today, thought of continuing
the **NLP Specialization** and learnt about **Transfer Learning In NLP** and explored few **language models**. <br> <br>

- **Transfer Learning In NLP :** It  means taking a language model that has already been trained on a lot of text and then adapting 
it for a specific task, like understanding sentiment or answering questions, to save time and improve accuracy. <br> <br>
  ![transferLearning](Day61_To_70/day61_transferLearning.png) <br> <br>
- **Self-Supervised task :** Unlabeled data is easy to get because it’s everywhere, like in articles, websites, and books, 
and doesn’t need people to label it. But labeled data is harder to find because it needs humans to go through and tag it, 
which takes a lot of time and effort. **Self-supervised learning** helps by allowing models to learn from this huge amount of unlabeled
data, creating their own tasks (like predicting missing words) to teach themselves and find patterns without needing human labels.
    <br> <br> ![selfSupervised](Day61_To_70/day61_selfSupervisedTask.png) <br> <Br>
- **Language Models summary :** <br> <br> ![LanguageModels](Day61_To_70/day61_ELMO2T5.png) <br> <br>

___

## Day 62

Learnt briefly about **BERT Pre-training** and **BERT FINE TUNING**. <br> <br>

- **BERT** is pre-trained using **Masked Language Modeling (MLM**), where random words in a sentence are masked, and the model predicts 
them using the surrounding context, and **Next Sentence Prediction (NSP)**, which trains the model to identify if one sentence logically 
follows another. These tasks enable **BERT** to understand both word-level and sentence-level relationships, which are fine-tuned on specific tasks using labeled data.
  <br> <br> ![finetuning](Day61_To_70/day62_bertFineTuning.png)

___

## Day 63

Completed the **NLP Specialization.** The remaining portion was about **T5 Model Architecture** and simple introduction to 
**Hugging Face**. <br> <br>

- **T5 Model Architecture :** The **T5 model** supports different architectures: the **Language Model (LM)**, **Prefix LM**, and the **encoder-decoder** 
structure. In a Language Model, the input and output tokens are processed sequentially, focusing only on previous tokens (causal attention).
**Prefix LM** combines bidirectional attention for the input and causal attention for generating the output. In T5’s main encoder-decoder setup,
the encoder uses bidirectional attention for full context understanding, while the decoder uses causal attention to generate output step by step. This flexibility allows T5 to handle diverse tasks like translation, summarization, and more.
 <br> <br> ![modelArchitectureT5](Day61_To_70/day63_T5Architecture.png) <br> ![multiTasking](Day61_To_70/day63_multiTaskingStrategy.png) 

___

## Day 64

Started exploring **OpenCV**. Went through the basics of **OpenCV** i.e. tasks like **reading** and **display8ing** images/vidos,
**resizing**, **dilation**, **erosion**, etc. <br> <br>

- **Reading and Displaying Images Videos** <br> <br> ![reading](Day61_To_70/day64_openCVReading.png) <br> <br>
- **Rescaling** <br> <br> ![Rescaling](Day61_To_70/day64_openCVRescaling.png) <br> <br>
- **Drawing** <br> <br> ![Drawing](Day61_To_70/day64_openCVDraw.png) <br> <br>
- **Basic Functions** <br> <br> ![BasicFunctions](Day61_To_70/day64_openCVBasicFunctions.png) <br> <Br>
___

## Day 65

Learnt to perform **transformations** like **translation**, **rotation** and **flipping**, **Contours** Detection and **Color Spaces**
in **OpenCV**. <br> <br>

- **Transformations** <br> <br> ![Transformations](Day61_To_70/day65_transformations.png) <br> <br>
- **Contours Detection** <br> Contours are the outlines or boundaries of objects in an image, like tracing around a shape.
    It might sounds like edges but edges are the sharp changes in brightness that highlight details but don’t always form complete shapes. <br> <br>
    ![Contours](Day61_To_70/day65_Contours.png) <br> <br>
- **Color Spaces** <br> <br> ![ColorSpaces](Day61_To_70/day65_colorSpaces.png) 

___

## Day 66

Learnt to perform **Splitting and Merging** of **Color Channels**, **Bitwise Operations** and **Masking**.
<br> <br>
- **Splitting And Merging Color Channels** <br> 
 It separates an image into its individual red, green, and blue components to 
analyze or manipulate them independently, while merging combines modified or individual channels back into a complete 
image for visualization or processing. <br> <br> ![splitMerge](Day61_To_70/day66_splitMergeColorChannels.png) <br> <br>
- **Blurring** <br> In OpenCV, **average blurring** applies a uniform kernel to average pixel values in a neighborhood, 
reducing noise but potentially losing edge details, while **GaussianBlur** uses a weighted kernel based on a Gaussian function, 
offering smoother results with less edge distortion. **Median blur** replaces each pixel value with the median of its neighborhood,
making it effective for salt-and-pepper noise, and **bilateral blur** preserves edges by combining spatial and intensity-based 
filtering to smooth regions without blurring edges. <br> <br> ![blurring](Day61_To_70/day66_blurring.png) <br> <br>
- **Bitwise Operations** <br> It manipulates pixel values at the binary level and are commonly used for tasks like masking, combining regions of interest, and performing logical operations on images.
    <br> <br> ![bitwiseOperations](Day61_To_70/day66_bitwise.png) <br> <br>
- **Masking** <br> It involves creating a binary mask to isolate or highlight specific regions of an image for focused processing, commonly used in tasks like region-based filtering, feature extraction, or selective analysis.
   <br> <br> ![Masking](Day61_To_70/day66_masking.png) <br> <br>

___

## Day 67

Learnt **Histogram** and other types of **Thresholding** and **Edge Detection** in **OpenCV**.
<br> <br>
- **Histogram** are used for analyzing pixel intensity distributions and performing tasks like contrast enhancement, thresholding, 
and object segmentation. <br> <br> ![hist](Day61_To_70/day67_histogram.png) <br> <br>
-  **Thresholding :** **Simple thresholding** requires a manually provided global threshold value for the entire image, while **adaptive thresholding** 
automatically calculates thresholds dynamically for different regions based on local pixel intensities.<br> <br> ![thresh](Day61_To_70/day67_threshold.png) <br> <br>
- **Edge Detection :** **Laplacian** detects edges using the second derivative to highlight rapid intensity changes, **Sobel** uses first derivatives with directional
filters to find edges in specific orientations, while **Canny** combines Gaussian smoothing, gradient calculation, and hysteresis thresholding for more
robust and accurate edge detection. <br> <br> ![edgeDetection](Day61_To_70/day67_edgeDetection.png) <br> <br>

___

## Day 68

Started exploring **Hugging Face**. Today just explored on what **Transformers** can do i.e. overview. <br> <br>

- ![TransformersOverview](Day61_To_70/day68_whatTranformersCanDO.png)

For the full notebook : [**Notebook**](https://github.com/iamshishirbhattarai/HuggingFace/blob/main/WhatTransformersCanDo.ipynb)

___

## Day 69

Got to explore **LlamaIndex**. Especially, focused on Learning to build **Agentic RAG** with **Llamaindex**.
<br> <br>

- **Router Query Engine :**  In LlamaIndex, it helps direct questions to the right set of data or search method. It ensures that each query gets the best possible 
answer by choosing the most relevant source to look in. <br><br> ![Router Query Engine ](Day61_To_70/day69_routerQueryEngine.png) <br> 

  Visit the full notebook here : [**Router Query Engine**](https://github.com/iamshishirbhattarai/Building-Agentic-RAG-With-LlamaIndex/blob/main/RouterQueryEngine.ipynb) <br> <br>

- **Tool Calling :** In LlamaIndex, it allows queries to trigger external tools, APIs, or functions dynamically, enabling more interactive and powerful responses. It helps integrate real-world actions,
 computations, or data retrieval into the LLM’s responses, making it more functional beyond just retrieving text. <br> <br> ![ToolCalling](Day61_To_70/day69_ToolCalling.png) <br> 
   Visit the full notebook here: [**Tool Calling**](https://github.com/iamshishirbhattarai/Building-Agentic-RAG-With-LlamaIndex/blob/main/ToolCalling.ipynb)


___

## Day 70

Read and implemented **Agent Reasoning Loop** and **Multi Documents Agent**.  <br> <br>

- An **agent reasoning loop** is an iterative process where an AI agent **analyzes a task, executes incremental steps, evaluates progress, and refines its actions** until a final solution is reached. <br> <br>
 ![agentReasoningLoop](Day61_To_70/day70_AgentReasoningLoop.png) <br>
 Visit the full notebook : [**AgentReasoningLoop**](https://github.com/iamshishirbhattarai/Building-Agentic-RAG-With-LlamaIndex/blob/main/AgentReasoningLoop.ipynb) <br> <br>

 - A **multi-document agent** is an AI system that **retrieves, processes, and reasons over multiple documents simultaneously** to generate informed responses or insights. <br><br>
 ![MultiDocumentAgent](Day61_To_70/day70_MultiDocumentsAgent.png) <br>
 Visit the full notebook : [**MultiDocumentAgent**](https://github.com/iamshishirbhattarai/Building-Agentic-RAG-With-LlamaIndex/blob/main/MultiDocumentAgent.ipynb)

 ___

 ## Day 71

 Explored **Llama-OCR** and **Llama-Parse**.

 - **LlamaOCR** is an OCR-based tool designed for raw text extraction from images, scanned documents, and handwritten text, without preserving structure. On the other hand, **LlamaParse** is an advanced document parser that extracts and structures text accurately from PDFs, Markdown, and other formatted documents, ensuring that headings, tables, and formatting are preserved. The key difference is that **LlamaOCR** focuses on character recognition, while **LlamaParse** ensures structured document extraction for AI applications.
   Attaching a short code snippet of each: <br> <br> **OCR** <br> ![LlamaOCR](Day71_To_80/day71_llamaOCR.png) <br> **Parse** <br> ![LlamaParse](Day71_To_80/day71_llamaParse.png)

___

## Day 72

Simply learnt about **Vector Databases** and especially learnt about **Qdrant** and its **Architecture**.

- A **vector database** is a specialized database optimized for storing, indexing, and querying high-dimensional vector embeddings, enabling fast similarity searches for applications like AI, recommendation systems, and natural language processing. <br> <br>
- **Qdrant** is an open-source vector database that helps store and search high-dimensional data quickly, making it useful for AI applications like recommendation systems and semantic search. <br> <br>
  **Qdrant's Architecture** <br><br> ![qdrant's Architecture](Day71_To_80/day72_QdrantArchitecture.png)
___

## Day 73



Practiced creating a simple query based Chatbot that answers based on the pdf provided. <br> <br>

- ![AIChatbot](Day71_To_80/day73_simpleChatbot.png)

____

## Day 74

Though surfacely I have been introduced to **RAG Pipeline** and has been already working on it, I thought of diving deeper 
and understand the fundamentals.

- **RAG Pipeline :** The **RAG pipeline** first processes and converts documents into a searchable format and stores them in a database. When a user asks a question, the system finds the most relevant information, combines it with the user’s question, and creates a **prompt**. This prompt is then sent to an AI model, which uses both the retrieved information and the question to generate a more accurate and informed response. <br> <br>
  ![ragPipeline](Day71_To_80/day74_ragPipeline.png) <br> <br>
- **Sentence Embeddings and Sentence BERT :** **Sentence embeddings** are numerical representations of sentences that capture their meaning in a way that allows for efficient comparison and retrieval in machine learning tasks. **Sentence-BERT (SBERT)** is a modification of BERT designed to generate high-quality sentence embeddings by fine-tuning BERT with a **Siamese network** structure, making it much faster and more effective for tasks like semantic similarity and search.
 <br> <br> ![sentenceEmbeddings](Day71_To_80/day74_sentenceEmbeddings.png) <br> ![sentenceBERT](Day71_To_80/day74_sentenceBERT.png)

 ___

 ## Day 75

 Continuing the **RAG Pipeline**, got to learn about the **HSNW** algorithm.

 - The **Hierarchical Navigable Small World (HNSW)** algorithm is an advanced method used in vector databases for fast and efficient nearest neighbor search. It builds a multi-layered graph structure where higher layers contain fewer nodes for quick navigation, allowing faster retrieval of similar vectors compared to brute-force methods. <br> <br> ![HSNW](Day71_To_80/day75_HNSW.png)

   In addition to this, I also took a course on **Understanding Prompt Engineering** from **DataCamp** and read a research paper [**Large Language Models as Data Preprocessors**](https://arxiv.org/pdf/2308.16361) where I basically learnt about the use of Prompts for Data Pre-processing. 

___

## Day 76

Had a short overview on getting started with **Qdrant**. Will dive in detail from tomorrow.

- ![startingQdrant](Day71_To_80/day76_startingQdrant.png)

___

## Day 77

I created a **collection** in **Qdrant** and upserted the embeddings. Although the code ran without error, nothing was found in the collection. Will see this tomorrow. <br>
Additionally, Looked over the best ways to **chunk** text for **RAG**.

- **Character/Token Based Chunking:** Splits text into fixed-length segments by counting characters or tokens, without regard to semantic boundaries.
<br>**Recursive Character/Token Based Chunking:** Applies a hierarchy of splitting rules, first using high-level delimiters and then recursively breaking down segments with lower-level token splits until each chunk meets a desired size.
<br>**Semantic Chunking:** Divides text at natural points where shifts in meaning or topic occur, ensuring that each chunk represents a coherent semantic unit.
<br>**Cluster Semantic Chunking:** Uses clustering algorithms on text embeddings to group semantically similar sentences or paragraphs into cohesive chunks.
<br>**LLM Semantic Chunking:** Leverages large language models to dynamically determine optimal chunk boundaries based on contextual and semantic cues. <br><br>
 ![evalReport](Day71_To_80/day77_evalTable.png)

___

## Day 78

Explored different types of **Retrievals** in **LangChain**. 

You can see all types of retrievals here in detail : [**LangChain Docs**](https://python.langchain.com/v0.1/docs/modules/data_connection/)

- ![**LangChain Retrieval**](Day71_To_80/day78_LangchainRetrievers.png) 

___

## Day 79


While exploring various other latest methods of **Chunking**, discovered about **Late Chunking**. Read the research paper, and got to know about this in detail. 
<br> 

- **Late chunking** is a method where a document is first tokenized and embedded as a whole before being dynamically chunked based on token embeddings. This approach preserves global context, ensuring that chunk embeddings retain interdependent relationships rather than treating chunks as isolated units.
  **Long Late Chunking** extends Late Chunking by introducing overlapping chunks, ensuring better context retention across chunks while maintaining efficient processing for long documents. <br> <br>
  ![Traditional Vs Late](Day71_To_80/day79_traditionalVsLateChunking.png)
  ![LateChunkingAlgo](Day71_To_80/day79_lateChunkingAlgo.png)
  ![LongLateChunkingAlgo](Day71_To_80/day79_longLateChunkingAlgo.png)  <Br>  <br>
  
   Research Paper : [**Late Chunking: Contextual Chunk Embeddings  Using Long-Context Embedding Models**](https://arxiv.org/pdf/2409.04701)

   ___

## Day 80

Completed **Fundamentals of Agents** from the **Hugging Face Agents Course**. 
Visit the details here : https://huggingface.co/learn/agents-course/en/unit0/introduction <br> <br>
- ![HuggingFaceUnitOne](Day71_To_80/day80_huggingFaceAgentsUnitOne.jpeg)

___

## Day 81

Completed **Unit 2** From the **Hugging Face Agents Course** that focuses on AI Agents Frameworks like **Smolagents** and **LlamaIndex**.
Already been familiar with **LlamaIndex** (you may visit the github repository) : [**Building-Agentic-RAG-With-LlamaIndex**](https://github.com/iamshishirbhattarai/Building-Agentic-RAG-With-LlamaIndex).

Along with this, dived deeper into LLMs through the **Andrej Karpathy's** youtube video: [**Deep Dive into LLMs like ChatGPT**](https://www.youtube.com/watch?v=7xTGNNLPyMI). <br> <br>
- ![DeepDiveIntoLLMs](Day81_To_90/day81_deepDiveIntoLLMs.png)

  _Image Derived from Andrej's resources_  (Resource Link: [**Click Here**](https://drive.google.com/file/d/1EZh5hNDzxMMy05uLhVryk061QYQGTxiN/view))

  ___

## Day 82

Dived deeper into **BPE (Byte-Pair Encoding)** and also did a simple implementation from scratch.


- **Byte-Pair Encoding (BPE)** is a method used to break words into smaller parts, helpful in natural language processing.  It starts by treating each character as a token and then merges the most frequent pairs of characters or tokens to form new subword units.  
This process repeats until a desired vocabulary size is reached.  
BPE helps handle rare or unknown words by breaking them into known pieces, improving efficiency.  
It also reduces the total number of tokens needed, saving memory and speeding up processing.
<br>

  ![bpe](Day81_To_90/day82_BPEImplementation.png)

  Visit the Notebook : [**BPE Implementation**](https://github.com/iamshishirbhattarai/Large-Language-Models/blob/main/simple_bpe_from_scratch.ipynb)


___



  