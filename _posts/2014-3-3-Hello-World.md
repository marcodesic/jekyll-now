---
layout: post
title: Neural Language Modeling From Scratch (Part 1)
mathjax: true
---

Language models assign probabilities to word sequences. Those three words that appear right above your keyboard on your phone that try to predict the next word you’ll type are one of the uses of language modeling. In the case shown below, the language model is predicting that “in”, “is” and “to” have a high probability of being the next word in the given sentence. Internally, for each word in its’ vocabulary, the language model computes the probability that it will be the next word, but the user only gets to see the top three most probable words.  

<div class="imgcap">
<img src="/images/lm/keyboard.png">
</div>

Language models are a fundamental part of many systems that attempt to solve hard natural language processing tasks such as machine translation and speech recognition. 

The first part of this post presents a simple feedforward neural network that solves this task. In the second part of the post, we will improve the simple model by adding to it a recurrent neural network (RNN). The final part will discuss two recently proposed techniques for improving RNN based language models, which are currently used to obtain state of the art results.


## A simple model

To begin we will build a simple model that given a single word taken from some sentence tries predicting the word following it.

_simple model, remove transpose_

We represent words using one-hot vectors: after giving a unique integer ID `n` to each word in our vocabulary, each word is represented as a one dimensional vector of the size of the vocabulary (`N`), which is set to `0` everywhere except for a single `1` at element `n`. 

The model can be seperated into two components:
* We start by **encoding** the input word. This is done by taking the one hot vector representing the input word, and multiplying it by a matrix of size `(N,200)` which we call the input embedding (`U`). This multiplication results in a vector of size `200`, which is also referred to as a word embedding. This embedding is a dense representation of the current input word. This representation is both of a much smaller size then the one-hot vector representing the same word, and also has some other interesting properties. For example, while the distance between every two words represented by a one-hot vectors is always the same, these dense representations have the property that words that are close in meaning will have representations that are close in the embedding space.

* The second component can be seen as a **decoder**. After the encoding step, we have a representation of the input word. We multiply it by a matrix of size `(200,N)`, which we call the output embedding (`V`).  The resulting vector of size `N` is then passed through the softmax function, normalizing its values into a probability distribution (meaning each one of the values is between `0` and `1`, and their sum is `1`). 

The decoder is a simple function that takes a representation of the input word and returns a distribution which represents the model's predictions for the next word. 

To train this model we use stochastic gradient descent, and the loss used is the cross-entropy loss. Intuitively, this loss measures the distance between the output distribution predicted by the model and the target distribution at every timestep. The target distribution at each iteration is a one-hot vector representing the current target word. 

For the `(input, target-output)` word pairs we use the Penn Treebank dataset which contains around 40K sentences from news articles. To generate word pairs for the model to learn from, we will just take every pair of neighbouring words from the text and use the first one as the input word and the second one as the target output word. So for example for the sentence `“The cat is on the mat”` we will extract the following word pairs for training: `(The, cat)`, `(cat, is)`, `(is, on)`, and so on. The vocabulary of the Penn Treebank dataset contains exactly `10,000` words. 

The metric used for reporting the performance of a language model is its perplexity on the test set. It is defined as- $$e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}}  $$, where $$p_{\text{target}_i}$$ is the probability given by the model to the target word at iteration 'i'. Perplexity is a decreasing function of the average log probability that the model assigns to the target word at every iteration. We want to maximize the probability that we give to the target word at every iteration, which means that we want to minimize the perplexity (the optimal perplexity is `1`).  

The perplexity for the simple model[^sg] is about `183` on the test set, which means that on average it assigns a probability of about $$ 0.005$$  to the target word in every iteration on the test set. Its much better than just a random guess (which would assign a probability of $$\frac {1} {N} = \frac {1} {10,000} = 0.0001$$ to the correct word), but we can do much better.

 
## Using RNNs to improve performance
The biggest problem with the simple model is that to predict the next word in the sentence, it only uses a single preceding word. If we could build a model that would remember even just a few of the preceding words there should be an improvement in its performance. To understand why adding memory helps, think of the following example: what words follow the word "drink"? You'd probably say that "coffee", "beer" and "soda" have a high probably of following it. If I told you the word sequence was actually "Cows drink", then you would completley change your answer.

We can add memory to our model by augmenting it with a [recurrent neural network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (RNN), as shown below.

_rnn model_



This model is just like the simple one, just that after encoding the current input word we feed the resulting representation (of size `200`) into a two layer [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), which then outputs a vector also of size `200` (at every timestep the LSTM also recieved a vector of size `200` representing it's previous state). Then, just like before, we use the decoder to convert this vector into a vector of probability values. (LSTM is just a fancier RNN that is better at remembering the past. Its "API" is identical to the "API" of an RNN- the LSTM at each time step an input and its previous state, and uses those two inputs to compute an updated state and an output vector[^api].)

Now we have a model that at each time step gets not only the current word representation, but also the state of the LSTM from the previous time step, and uses this to predict the next word. The state of the LSTM is a representation of the previously seen words (note that words that we saw recently have a much larger impact on this state then words we saw a while ago). 

As expected, performance improves and the perplexity of this model on the test set is about `114`. An implementation of this model[^zaremba], along with a detailed explanation, is available in [Tensorflow](https://www.tensorflow.org/tutorials/recurrent).

## The importance of regularization. 
`114` perplexity is good but we can still do much better. In this section I'll present some recent advances that improve the performance of RNN based language models. 

### Dropout

We could try improving the network by increasing the size of the embeddings and LSTM layers (until now the size we used was `200`), but soon enough this stops increasing the performance because the network overfits the training data (it uses its increased capacity to remember properties of the training set which leads to inferior generalization, i.e. performance on the unseen test set). One way to counter this, by regularizing the model, is to use dropout. 

The diagram below is a visualization of the RNN based model unrolled across three time steps. `x` and `y` are the input and output sequences, and the grey boxes represent the LSTM layers. Vertical arrows represent an input to the layer that is from the same time step, and horizontal arrows represent connections that carry information from previous time steps. 

_regular_

We can apply dropout on the vertical (same time step) connections:
_vanilla dropout_

The arrows are colored in places were we apply dropout. We use different dropout masks for the different connections (this is indicated by the different colors in the diagram). 

Applying dropout to the recurrent connections harms the performance, and so in this initial use of dropout we use it only on connections within the same time step. Using two LSTM layers, with each layer containing `1500` LSTM units, we acheive a perplexity of `78`. 

A recent [dropout modification](https://arxiv.org/abs/1512.05287) solves this problem and improves the model's performance even more (to `75` perplexity) by using the same dropout masks at each time step. 

_variational do_



### Weight Tying 

An RNN based language model consists of three components: the input embedding (the "encoder"), the RNN (the "processor"), and an output embedding, which in conjuction with a softmax layer is the "decoder". (The RNN component can be seen as a "processor" because at every time step it recieves a representation of the current word and a representation of all words seen until now (the previous hidden state) and outputs a vector representing its belief about the next word). 

The input embedding and output embedding have a few properties in common. The first property they share is that they are both of the same size (in our RNN model with dropout they are both of size `(10000,1500)`). 

The second property is that in the input embedding, words that have similar meanings are represented by similar vectors (similar in terms of [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity#Definition)). This is because the model learns that it needs to react to similar words in a similar fashion (the words that follow the word "quick" are similar to the ones that follow the word "rapid").

This also occurs in the output embedding. The output embedding recieves a representation of the "processor"'s belief about the next output word (the output of the RNN) and has to transform this into a distribution. Given the representation from the "processor", the probability that the decoder assigns a word depends mostly on its representation in the output embedding (the probability is exactly the softmax normalized dot product of this representation and the output of the "processor"). 

Because the model would like to, given the RNN output, assign similar probability values to similar words, similar words are represented by similar vectors. (Again, if, given a certain RNN output, the probability for the word "quick" is relatively high, we would also expect the probability for the word "rapid" to be relatively high).
<be consistent with "rnn output"/ "processor output">

These two similarities lead us to propose a very simple method to lower the model's parameters and improve its performance. We simply tie its input and output embedding (i.e. we set U=V, meaning that we now have a single embedding matrix that is used both as an input and output embedding). This reduces the perplexity of the RNN model that uses dropout to `73`, and its size is reduced by more than 20%. 


Why does weight tying work?
Two reasons:
The perplexity of the vanilla RNN language model on the test set is XX. The same model achieves <YY> perplexity on the training set. So the model performs much better on the training set then it does on the test set. This means that it has started to learn certain patterns or sequences that occur only in the train set and do not help the model to generalize to unseen data. One of the ways to counter this overfitting is to reduce the models ability to 'memorize' by reducing its capacity (number of parameters). By applying weight tying, we remove a large number of parameters. 

The second reason is a bit more subtle. In our paper we show that the word representations in the output embedding are of much higher quality than the ones in the input embedding. This is shown using embedding evaluation benchmarks such as Simlex999<link>. In the weight tied model, because the tied embedding's parameter updates at each training iteration are very similar to the updates of the output embedding in the untied model, the tied embedding performs similarly to the output embedding of the untied model. So in the untied model, we use a single high quality embedding matrix in two places in the model. This contributes to the improved performance of the tied model. (Read the paper for the full explanation) <-- foot note


To summarize, we showed how to improve a very simple feedforward neural network language model, by first adding an RNN, and then adding variational dropout and weight tying.

In recent months, we've seen further improvements to the state of the art in RNN language modeling. The current state of the art results are held by <blunsom> and <merity> . These models take use most, if not all, of the methods shown above, and extend them by using better optimizations techniques, new regularization methods, and by finding better hyperparameters for existing models. 

Feel free to ask questions in the comments bellow. 
  
  
  
***
  



[^sg]: This model is the skip-gram word2vec model presented in [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
[^api]: For a detailed explanation of this watch [Edward Grefenstette's "Beyond Seq2Seq with Augmented RNNs" lecture.](http://videolectures.net/deeplearning2016_grefenstette_augmented_rnn/) 
[^zaremba]: This model is the small model presented in [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329).
