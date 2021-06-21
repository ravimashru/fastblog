---
layout: post
description: Reading notes taken when going through chapter 1 of fastbook
categories: [fastbook, reading-notes]
title: Reading Notes - Ch. 1 - Fastbook
---

> Deep learning is a computer technique to extract and transform data... by using multiple layers of neural networks.

> The layers are trained by algorithms that minimize their errors and improve their accuracy.

What you don't need to get great results with deep learning:

- Lots of math (high school math is sufficient)
- Lots of data (record breaking results obtained with <50 items of data)
- Lots of expensive computers (you can get what you need to obtain state-of-the-art results for free)
- A PhD

Examples of places where deep learning is the best in the world

- Natural Language Processing
	- [Creating Images from Text](https://openai.com/blog/dall-e/)
	- [Text Summarization](https://deepai.org/machine-learning-model/summarization)
	- [Smart Compose in Gmail](https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html)
- Computer vision
	- [Image classification](http://destiny.liacs.nl/)
	- [Image captioning](https://milhidaka.github.io/chainer-image-caption/)
	- [Autonomous vehicles](https://www.youtube.com/watch?v=9e2x4dDRB-k&ab_channel=FredrikGustafsson)
- Medicine
	- [Diabetic retinopathy detection](https://health.google/for-clinicians/ophthalmology/)
- Image generation
	- [Image Inpainting](https://www.nvidia.com/research/inpainting/)
	- [Increasing image resolution](https://letsenhance.io/)
	- [Converting images to art in the style of famous artists](https://fluxml.ai/experiments/styleTransfer/)
- Recommendation systems
	- Amazon
	- Netflix
- Playing games
	- [AlphaGo](https://www.youtube.com/watch?v=WXuK6gekU1Y)
	- [Breakout](https://www.youtube.com/watch?v=TmPfTpjtdgg&t=72s&ab_channel=DeepMind)
- Robotics
	- [Clearing an obstacle course](https://www.youtube.com/watch?v=fRj34o4hN4I&t=54s&ab_channel=BostonDynamics)
	- [Dancing!](https://www.youtube.com/watch?v=fn3KWM1kuAw&ab_channel=BostonDynamics)


The fastai library adds higher-level functionality on top of PyTorch. However:

> ...it doesn't really matter what software you learn, because it only takes a few days to switch from one library to another. What really matters is learning the deep learning foundations, and techniques properly.

> You should assume that whatever specific libraries and software you learn today will be obsolete in a year or two.

## Machine learning vs regular programming

With regular programming, we write down for the computer the exact steps necessary to complete a task. But when trying to recognize an object in a picture, we don't know what steps we take - it all happens in our brain without us being consciously aware of it!

The basic idea of machine learning (according to Arthur Samuel) is to show the computer examples of the problem to solve and let it figure out how to solve it itself, instead of outlining the exact steps to be followed to solve the problem.

How he described his idea:

> Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.

The key ideas in this statement are:

1. We can assign "weights" that would determine how the program reacts to different inputs
2. For any given weights, there is an automatic means of measuring how well those weights allow the program to perform
3. There is a mechanism that improves this performance by making the necessary adjustments to the assigned weights


![]({{ site.baseurl }}/images/ml.png)

There are a few minor changes in modern terminology:

- The program that describes the functional form of the model is now called the model "architecture"
- Weights are now referred to as model "parameters"
- The results of the model are called "predictions"
- The measure of performance is called the "loss"
- The data consists of two parts: the "independent variable" that is used to calculate predictions and the correct "labels" that are used with the model predictions to calculate the loss

![]({{ site.baseurl }}/images/ml-today.png)

It's hard to understand why a deep learning model makes a particular prediction because it is hard to determine what parts of the input the different layers are using to pass information to the next layers, and deep learning models have several such layers. However, there are techniques we can use to determine which parts of the input a deep learning model is focusing in while making a predictions (e.g. [Grad-CAM](http://gradcam.cloudcv.org/))

The universal approximation theorem shows that a neural network can, in theory, solve any problem to any level of accuracy.

Stochastic Gradient Descent (SGD) is a general way to update the weights of a neural network so that its performance on any task improves.

## Limitations of machine learning

- A model cannot be created without data
- A model can learn how to make predictions based on only patterns seen in the data used to train it
- A trained model can only make predictions, and not provide recommended actions
- We don't just need input data, we need correct labels for that data too


Machine learning models can be negatively affected based on how they interact with their environment. Example: predictive policing.

- Model created based on where arrests have been made in past
- As a result, model is not predicting crime but reflecting biases in existing policing processes
- Law enforcement officers try to use model to decide where to focus effort, resulting in more arrests in those areas
- Data of additional arrests used to retrain future versions of model

This is a "positive feedback loop" - model is used more => data becomes more biased => model becomes even more biased, and so on.

Another example of a feedback loop:

- Conspiracy theorists and extremists tend to watch more online video content than average.
- A video recommendation system that uses this data will be biased towards recommending more of such content
- As a result, such users increase their video consumption leading to more such videos being recommended.

Transforms: code that is applied to data before being fed into the model during training.

 - `item_tfms`: applied to each item (on CPU)
 - `batch_tfms`: applied to entire batch of items (on GPU)

Classification - predict class or category
Regression - predict one or more numeric quantities

## Training and Validation set

Data that you hold out during training so that you can use it later to evaluate the performance of the model on data it has never seen before. The remaining data that is used to train the model is called the training set.

If not specified, fastai uses a default validation set of size 20%. fastai always shows model's accuracy using only validation set.

We need to use a validation set to measure the model's performance so that we can be sure it isn't overfitting - memorizing the data in the training set instead of finding underlying patterns in the data.

> Overfitting is the single most important and challenging issue

However, use methods to avoid overfitting only after confirming that overfitting is occurring. Using overfitting avoidance techniques when you don't need to leads to a model that is less accurate than what could be achieved.

## Model architecture

Picking an architecture isn't a very important part of the process, not something you spend too much time on. There are some standard architectures that work great most of the time.

## Loss vs. Metric

A metric is a function that measures the quality of the model's predictions.

A loss function is used to define a measure of performance that can be used by a training algorithm to automatically update the model's weights.

Loss functions are meant for the training algorithm (e.g. SGD) and metrics are meant for humans - they should be easy to understand.

Sometimes, the loss may be a suitable metric but not always.

## Pretrained models

> A model that has weights that have already been trained on another dataset is called a **pretrained model**

Always try to start with pretrained models since they already have some capabilities before you even train them on your dataset.

When using a pretrained model, last layer is removed because it is very specific to task that model was initially trained on. It is replaced with new layers with randomized weights that fits the task currently trying to be solved. This part of the network is called the **head**.

Using pretrained models allows us to:
- train more accurate models
- train models more quickly
- train models using less data
- train models with less time and money.

## Transfer Learning

Transfer learning is when you take a pretrained model and use it for a task different from the one it was originally trained for.

Current challenges:
 - Transfer learning is highly understudied and therefore few domains have pretrained models available
 - It is not yet well understood how to use transfer learning for tasks such as time series analysis

**Fine-tuning** is the process of adapting a pretrained model for a new dataset. You update the weights of the model using the new dataset.

The `fine_tune` method in fastai does the following:

1. Use one epoch to fit parts of the model that get the new head working correctly with the new dataset
2. Updates parameters of the entire models. Weights of later layers (especially the head) are updated much faster than earlier layers

## Non-Image Tasks

Image recognizers are capable of performing non-image tasks.

Examples of projects where image recognizers were used

- Convert environmental sounds to spectrogram
- Convert time series data to image using  Gramian Angular Difference Field
- Draw time-series data of mouse movements (position, speed, acceleration using different colors) and clicks
- Convert binary file to image where each byte is converted to color pixel in grayscale


## Tabular data

Data in the form of a table. A tabular model tries to predict the values in one column of the table based on values in the other columns.

When training a model on tabular data, we have to specify which columns are categorical (contain values that are from a set of discrete choices) and which ones are continuous (numbers representing a quantity).

Generally, there are no pretrained models available for tabular modeling tasks. However, some organizations may have created some for internal use.

## Rapid prototyping

THe world's top practitioners do their experimentation and prototyping with just a subset of their entire dataset. Once they have a good understanding of what they have to do, they start using the full dataset.

Using a subset of data makes training models quicker and allows you to iterate faster.

## Validation and test set

Already discussed: importance of training and validation set.

We rarely train model just once. We explore many versions of a model by making choices for **hyperparameters** like network architecture, learning rates, data augmentation strategies.

We look at model performance on validation data while exploring hyperparameter values. As a result, the model is at risk of overfitting validation data through human trial and error and exploration.

The test set is a subset of data apart from training and validation data. It is held back when trying to improve model by tuning hyperparameters and used only at the end to evaluate the final performance of the model.

> ...if you have very little data, you may need just a validation set - but generally it's best to use one if at all possible

Validation and test sets must be representative of new data the model will see in the future.

For time series data, since the model will be making predictions in the future having seen only past data, the validation set can be a continuous section of the dataset with the latest dates instead of random points. Choosing random points means the model trains by seeing data that comes before and after the date it needs to make the prediction, however when the model is actually being used it will only have knowledge of past data.

Also consider the qualitative differences in data used to train model and data that the model will be seeing in production. Example: training an image recognition model to see if a driver is distracted may see people while running in production that it has never seen during training. So keep images in validation set that are of people the model has never seen during training.

