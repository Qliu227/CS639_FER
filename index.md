# Facial Emotion Recognition Using Deep Convolutional Neural Network
Team members
Shanchao Liang (sliang53@wisc.edu)
Zhaoyang Li (zli2344@wisc.edu)
Qitong Liu (qliu227@wisc.edu)
Shirley Liu (rliu326@wisc.edu)


## 1. Introduction
Facial expression is an important communication tool for the human being, through which people can recognize others’ mental status, such as happiness, anger, sorrow, etc.[1] Among various facial emotion recognition (FER) methods, camera image has become a focused area because of its widespread application in daily life, such as Virtual Reality, advanced driver assistance systems (ADASs), education, and healthcare.[2] In light of this, the FER aroused our interest to study and implement some techniques.

Facial emotion recognition (FER) is achieved either by measuring the changes of various body parameters, such as measuring heart rate, eye activity, electroencephalography, etc or by analyzing the facial image.[1] And the latter has gained popularity because of the abundant and cost-effective computational resources. The research studies the FER based on input from the camera can be divided into two groups, conventional methods and approaches using neural networks.[1] The conventional FER method is based on hand-engineered features to identify facial emotion.[4] The typical process includes detecting the region of interest, extracting features and then using a pre-trained classifier to recognize the facial emotion, as shown in Fig.1.[1] The popular classifier includes SVM, random forest, and kNN. In contrast, the neural network uses the whole image as the input and it is processed by artificial neurons. It has emerged as a general approach for computer vision with the availability of big datasets. The classical CNN convolves the input image to generate feature maps, which are concatenated and then used for classification (see Figure 2).[1] Among published studies, Kim et al. proposed a hybrid model consisting of a CNN and long short-term memory (LSTM), and it has the highest performance with an accuracy of 78.61%. [3]

## 2. Methods
In this project, we will build a deep convolutional neural network (CNN) based on the model proposed by 
Aayush Mishra (https://www.kaggle.com/aayushmishra1512/emotion-detector), which consists of five CNN modules. The model contains about 4.5 million parameters. The batch normalization and dropout techniques are used to make the model robust. The detailed structure is shown in Fig.3. 

Figure 1. Conventional FER method. [1]

Figure 2. Deep neural networks-based FER approach. [1]

Figure 3. The architecture of CNN

Several basic components and techniques adopted in the model are briefly reviewed. 
Convolution: The convolution kernel is used to detect edges and thus, it is also called the filter. The convolution kernel has a locality attribute, and it focuses on local features. For example, the essence of edge detection with the Sobel operator is to compare the similarity of adjacent pixels of images.
Activation: In the biological sense of neurons, only when the weighted sum of the signals transmitted by the previous dendrites is greater than a certain threshold, the subsequent neurons will be activated. Similarly, when the output cannot reach a certain standard or the feature in a certain area is very weak, then the feature intensity output should be 0, which is achieved by the activation function. Thus, areas not related to the feature will not affect the training of the feature extraction method. Relu is the most commonly used activation function, which was employed in this project.
Pooling:[6] Pooling is a nonlinear downsampling method. After obtaining image features through convolution, these features are used for classification. Using all the extracted feature data to train the classifier is possible, but this usually results in a tremendous calculation. Therefore, the dimensionality of the convolutional features should be reduced by the maximum pool sampling method after acquiring the convolutional features of the image. The convolution feature is divided into several n×n disjoint regions, and the maximum (or average) feature in these regions is used to represent the convolution features after dimensionality reduction. These reduced-dimensional features are easier to classify. Maximum pool sampling reduces the computation complexity from intermediates layers, and these pooled units have translation invariance.
Batch Normalization:[7] batch normalization is a preprocessing technique that is achieved before each layer of the network so that the activation data is standardized before the training starts. The network that uses batch normalization is more robust to bad initial values.
Dropout: In some machine learning models, if the model has too many parameters and too few training samples. Dropout can effectively solve the overfitting problem by letting some hidden nodes fail randomly. The Dropout method enables the neuron to be activated with the probability of the hyperparameter p to set to 0. In this way, no matter how large the weight is updated, the weight will not be too large. In addition, the algorithm can also use a relatively large learning rate to speed up the learning speed so that the algorithm can search for better weights in a broader weight space without worrying about excessive weights.
### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Qliu227/CS639_FER/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
