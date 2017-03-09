---
layout: post
title: You're up and running!
mathjax: true
---

Language models assign probabilities to word sequences. Those three words that appear right above your keyboard on your phone and try to predict the next word you’ll type are one of the uses of language modelling. In the case shown below, the language model is predicting that “in”, “is” and “to” have a high probability of being the next word in the given sentence. Internally, for each word in its’ vocabulary, the language model computes the probability that it will be the next word, but the user only gets to see the top three most probable words.  


Language models are a fundamental and critical part of many systems that attempt to solve solve hard natural language processing tasks such as machine translation and speech recognition. 

The first part of this post presents a simple neural network that solves this task. In the second part of the post, we will improve the simple model by adding to it a recurrent neural network. The final part will discuss further ways this model can be enhanced. 


## A simple model

To begin we will build a simple model that given a single word taken from some sentence tries predicting the word following it.

We represent words using one-hot vectors. After giving a unique integer ID `n` to each word in our vocabulary, each word is represented as a one dimensional vector of the size of the vocabulary (`V`), which is set to `0` everywhere except for a single `1` at element `n`. 


The model takes the one hot vector representing the input word, multiplies it by a matrix of size `(V,200)` which we call the input embedding. This results in a vector of size `200`, which we then multiply by a matrix of size `(200,V)`, which we call the output embedding. The resulting vector of size `V` is then passed through the softmax function, normalizing its values into a probability distribution (meaning each one of the values is between `0` and `1`, and their sum is `1`). 
To train this model we use SGD, and the loss used is the negative log likelihood loss. For the (input, output) word pairs we use the Penn Treebank dataset which contains around 40K sentences from news articles. To generate word pairs for the model to learn from, we will just take every pair of neighbouring words from the text and use the first one as the input and the second one as the target output. So for example for the sentence `“The cat is on the mat”` we will extract the following word pairs for training: `(The, cat)`, `(cat, mat)`, `(cat, is)`, and so on. The vocabulary of the Penn Treebank dataset contains exactly `10,000` words. 

The metric used for reporting the performance of a language model is its perplexity on the test set. The perplexity is defined as- \\(e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}}  \\), where p_target_i is. This is a decreasing function of the average log probability that the model assigns to the correct target word at every iteration (so lower is better). The perplexity for the simple model is 183 on the test set, which means that on average it assigns a probability of about 0.005 to the correct word. Its much better than just a random guess (which would assign a probability of $1/V = 1/10,000 = 0.0001 to the correct word), but we can do much better.


###Using RNNs to improve performance
The biggest problem with the simple model is that to predict the next word in the sentence, it only uses a single preceding word. If we could build a model that would look at all preceding words there should be an improvement in its performance. We can accomplish this by augmenting the network with a recurrent neural network, as shown below.





This model is just like the simple one, just that after the input embedding operation we feed the resulting vector of size 200 into a 2 layer LSTM, which then outputs a vector also of size 200. Then, just like before, we multiply this vector by the output embedding and then apply the softmax function.


Now we have a model that at each timestep gets not only the current word, but also the state of the LSTM layers from the previous timestep, and uses this to predict the next word. This state encodes all previously seen words (in an opaque and uninterpretable way). As expected, performance improves and the perplexity of this model on the test set is about 114. 
