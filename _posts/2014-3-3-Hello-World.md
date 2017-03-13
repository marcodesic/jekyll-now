---
layout: post
title: You're up and running!
mathjax: true
---

Language models assign probabilities to word sequences. Those three words that appear right above your keyboard on your phone that try to predict the next word you’ll type are one of the uses of language modelling. In the case shown below, the language model is predicting that “in”, “is” and “to” have a high probability of being the next word in the given sentence. Internally, for each word in its’ vocabulary, the language model computes the probability that it will be the next word, but the user only gets to see the top three most probable words.  


Language models are a fundamental part of many systems that attempt to solve hard natural language processing tasks such as machine translation and speech recognition. 

The first part of this post presents a simple feedforward neural network that solves this task. In the second part of the post, we will improve the simple model by adding to it a recurrent neural network. The final part will discuss further ways this model can be enhanced. 


### A simple model

To begin we will build a simple model that given a single word taken from some sentence tries predicting the word following it.

We represent words using one-hot vectors: after giving a unique integer ID `n` to each word in our vocabulary, each word is represented as a one dimensional vector of the size of the vocabulary (`V`), which is set to `0` everywhere except for a single `1` at element `n`. 

The model can be seperated into two components:
* We start by **encoding** the input word. This is done by taking the one hot vector representing the input word, and multiplying it by a matrix of size `(V,200)` which we call the input embedding. This multiplication results in a vector of size `200`, which is also referred to as a word embedding. This embedding is a dense representation of the current input word. This representation is both of a much smaller size then the one-hot vector representing the same word, and also has some other interesting properties. For example, while the distance between every two words represented by a one-hot vectors is always the same, these dense representations have the property that words that are close in meaning will have representations that are close in the embedding space.

* The second component can be seen as a **decoder**. After the encoding step, we have a representation of the input word. We multiply it by a matrix of size `(200,V)`, which we call the output embedding.  The resulting vector of size `V` is then passed through the softmax function, normalizing its values into a probability distribution (meaning each one of the values is between `0` and `1`, and their sum is `1`). 
The decoder is a simple function that takes a representation of the input word and returns a distribution which represents the model's predictions for the next word. 

To train this model we use stochastic gradient descent, and the loss used is the cross-entropy loss. Intuitively, this loss measures the distance between the output distribution predicted by the model and the target distribution at every timestep. The target distribution at each iteration is a one-hot vector representing the current target word. 

For the `(input, target-output)` word pairs we use the Penn Treebank dataset which contains around 40K sentences from news articles. To generate word pairs for the model to learn from, we will just take every pair of neighbouring words from the text and use the first one as the input word and the second one as the target output word. So for example for the sentence `“The cat is on the mat”` we will extract the following word pairs for training: `(The, cat)`, `(cat, is)`, `(is, on)`, and so on. The vocabulary of the Penn Treebank dataset contains exactly `10,000` words. 

The metric used for reporting the performance of a language model is its perplexity on the test set. It is defined as- $$e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}}  $$, where $$p_{\text{target}_i}$$ is the probability given by the model to the target word at iteration 'i'. Perplexity is a decreasing function of the average log probability that the model assigns to the target word at every iteration. We want to maximize the probability that we give to the target word at every iteration, which means that we want to minimize the perplexity (the optimal perplexity is `1`).  
The perplexity for the simple model is about `183` on the test set, which means that on average it assigns a probability of about $$ 0.005$$  to the target word in every iteration on the test set. Its much better than just a random guess (which would assign a probability of $$\frac {1} {V} = \frac {1} {10,000} = 0.0001$$ to the correct word), but we can do much better.


### Using RNNs to improve performance
The biggest problem with the simple model is that to predict the next word in the sentence, it only uses a single preceding word. If we could build a model that would remember even a few of the preceding words there should be an improvement in its performance. To understand why adding more words helps, 
We can accomplish this by augmenting the network with a recurrent neural network, as shown below.





This model is just like the simple one, just that after the input embedding operation we feed the resulting vector of size 200 into a 2 layer LSTM[colah], which then outputs a vector also of size 200. Then, just like before, we multiply this vector by the output embedding and then apply the softmax function.


Now we have a model that at each time step gets not only the current word, but also the state of the LSTM from the previous time step, and uses this to predict the next word. This state encodes the previously seen words (note that words that we saw recently have a much larger impact on this state then words we saw a while ago)[^1]. As expected, performance improves and the perplexity of this model on the test set is about 114. [link to https://www.tensorflow.org/tutorials/recurrent]



[^1]: Some *crazy* footnote definition.

