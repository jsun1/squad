[Paper](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report219.pdf)

# Stanford Question-Answering AI

The task of question answering (QA) requires language comprehension and modeling complex interaction between the context and the query [1]. Recurrent models primarily use recurrent neural networks (RNNs) to process sequential inputs, and attention component to cope with long term interactions [2]. However, recurrent QA models have issues recovering from local maxima due to the single-pass nature of the decoder step. To address this, we implemented a model based on Dynamic Coatention Network (DCN) that incorporates a dynamic decoder that iteratively predicts the answer span [3]. To improve the model efficiency, we implemented a transformer based recurrency-free model (QANet), which consists of a stack of encoder blocks including self-attention and convolutional layers. On the Stanford Question Answering Dataset ([SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)), our best QANet based model achieved 68.76 F1 score and 65.081 Exact Match (EM) on the dev set and 66.00 F1 and 62.67 EM on the test set.

## Approach

### DCN overview
The DCN model exhibits a modular architecture consisting of 4 main layers: embedding layer, document and question encoder, contention encoder and dynamic pointing decoder as presented in the dynamic coattention network (DCN) model [3]. The contention encoder captures the interaction between the question and the context and a dynamic pointing decoder that iteratively estimates the answer span. 

### Embedding layer (character embeddings)
As described in Seo et al 2016 [1], we adopt the standard technique to obtain the embedding of each word w by concatenating its word embedding and character embedding. We used the pretrained GloVe vectors for word embedding with dimension of 300 [13]. We implemented character level embedding using a 1D convolution with kernel size 3 and character embedding dimension of 100, with a maxpool layer. Experiments were also performed with a kernel size of 5. We experimented with word embeddings only, character embeddings only, and word plus character embeddings. The concatenation of the character and word embedding is fed into to a two layer Highway Network [14]. For our experiment we used d = 128 as the dimension of the embedding vector.

_One encoding block_

<img width="564" alt="Screen Shot 2022-08-13 at 5 36 18 PM" src="https://user-images.githubusercontent.com/3321825/184517805-7e725d4b-7107-4261-a53c-f7190573b77e.png">

### Document and question encoder
In this layer, we generate encoding representation of questions and contexts. Using an LSTM, the paper defines context encoding matrix C and question encoding matrix Q.

<img width="712" alt="Screen Shot 2022-08-13 at 5 34 00 PM" src="https://user-images.githubusercontent.com/3321825/184517774-39475a3b-2702-431a-92ab-2188fe1c842a.png">

Note a non-linear projection layer on top the question encoding is introduced to allow variation between the question encoding space and the context encoding space. Also, learnable sentinel vectors which allow model to not attend to any words are added to C and Q respectively.

### Coattention encoder

The coattention encoder learns co-dependant representations of the question and the document. We have implemented the encoder as described in the DCN paper [3]. We first compute L, will be used to calculate attention distribution in both direction, across the document for each word in the question (A<sup>Q</sup>) and across the question for each word in the context (A<sup>C</sup>). The question-to-context attention output (S<sup>Q</sup>) and coattention output (S<sup>C</sup>) are computed as following:

<img width="863" alt="Screen Shot 2022-08-13 at 5 48 18 PM" src="https://user-images.githubusercontent.com/3321825/184517977-9c4d854e-4a0d-4939-a08f-91751e1316d4.png">

We then pass the resulting hidden states U known as coattention encoding to the decoder module to predict answer span.

### QANet overview
We have implemented the QANet model, which is is a feedforward model consisting of convolution and self attention [2]. The QANet model consists of five layers, including an embedding layer, an embedding encoder layer, an attention layer, a model encoder layer and an output layer. The main module of QANet model is an Encoder Block, a stack of following building blocks: positional encoding, residual connection, layer normalization, a stack of convolution layers, multi-head attention layer [15] and feed forward layer.

### Embedding encoder layer
We feed the output of embedding layer W<sup>C</sup> and W<sup>Q</sup> into a single encoder block. The encoder block for context words and question words share the same weights. We used the configuration suggested by the original QANet model for our encoder block. The kernel size is 7, the number of filters is 128, the number of attention heads is 8 and the number of convolution layers within a block is 4.

### Term frequency and part-of-speech tags
Furthermore for additional input features in the embedding layer, as described in Chen et al 2017 [16], we added normalized term frequency and part-of-speech tags for each word. The normalized term frequency represents the number of times a word appears in the context or question (represented as a normalized float), and the part-of-speech tag is represented as a one-hot encoding over the 45 different part-of-speech tags. The part-of-speech tag is gathered from the nltk Python library, using the UPenn tagset.


## Experiments
### Data
The official Stanford Question Answering dataset SQUAD 2.0 will be used as the dataset [4]:

Context | Question | Ground truth answers
---|---|---
The further decline of Byzantine state-of-affairs paved the road to a third attack in 1185, when a large Norman army invaded Dyrrachium, owing to the betrayal of high Byzantine officials. Some time later, Dyrrachium—one of the most important naval bases of the Adriatic—fell again to Byzantine hands. | When did the Normans attack Dyrrachium? | 1185, in 1185, 1185

### Experimental details
We implemented the DCN and QANet models. In running our experiments, we tried to respect model configurations suggested in the original papers [3], [2]. For both models we used pre-trained GloVe word vectors for word embedding, and implemented character level embedding to better handle out-of-vocabulary words. The dimension of the character embedding vector is set to 64.

For the DCN model, we have randomly initialized all LSTMs weights and set initial state to zero. For the dynamic decoder layer, we set number integration to 4 and maxout pool size to 16 as suggested in the original paper[3]. We apply dropout to regularize the network and ADAM optimizer and constant learning rate a = 0.5 to preform SGD to minimize the loss function during training. We used batch size of 64, hidden size of 100 with dropout rate of 0.2 and trained the model for 30 epochs which took around 36 hours. 

For the QANet model, we set the hidden size and the convolution filter size to 128. The embedding and modeling encoders have 4 and 2 convolution layers with kernel size 7 and 5, respectively. As suggested in the original paper, we use a learning rate warm-up scheme with an inverse exponential increase from 0.0 to 0.001 in the first 1000 steps. We used batch size of 16, hidden size of 128 with dropout rate of 0.01 and trained the model for 30 epochs taking around 13 hours. 

_Character-level embeddings performance_
<img width="2547" alt="Screen Shot 2021-02-25 at 2 40 14 PM" src="https://user-images.githubusercontent.com/3321825/184517522-b56119ac-f808-4860-92e2-509c120a3431.png">

_Performance of various techniques_
<img width="1408" alt="Screen Shot 2021-03-18 at 8 23 13 PM" src="https://user-images.githubusercontent.com/3321825/184517534-86e27efb-d7a8-4e9d-ae6e-50979e9543bc.png">

## Results

We analyzed the performance of our models and its ablations on the SQUAD 2.0 development set as illustrated in the table below. Character embeddings improved the EM and F1 scores, whereas the additional input features sped up performance gains during training while only modestly improving overall performance.
The dynamic coattention network performed roughly the same as the baseline with character embeddings, while the QANet transformer model performed at a much higher level of accuracy than the other models.

_Experimental results on the development set_

Model | EM | F1
---|---|---
Baseline | 55.60 | 59.24
Character embeddings | 59.67 | 63.31
Word + character embeddings | 59.49 | 62.75
Baseline | 60.23 | 63.54
Baseline | 59.28 | 62.62
Baseline | 55.73 | 59.72
Baseline | 58.28 | 61.82
Baseline | 65.08 | 68.76

## Conclusion

In conclusion, we do not find significant performance gains for the dynamic coattention network on the question answering domain, although we do find that the QANet transformer based model achieves increased performance of an EM score of 62.671 and F1 score of 66.005 on the test leaderboard. Furthermore, additional input features like character embeddings and normalized term frequency helps the model achieve a significant 4.3 point F1 score increase.

## References

1. Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. Bidirectional attention flow for machine comprehen- sion. In https: //arziv. org/abs/ 1611. 01603, 2016.
2. M.-T. Luong R. Zhao K. Chen M. Norouzi A. W. Yu, D. Dohan and Q. V. Le. Qanet: Combining local convolution with global self-attention for reading comprehension. In https: //arziv. org/abs/ 1804. 09541, 2018.
3. Caiming Xiong, Victor Zhong, and Richard Socher. Dynamic coattention networks for question answering. In https: //areiv. org/abs/ 1611. 01604, 2016.
4. Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In https: //arxiv. org/abs/ 1606. 05250, 2016.
5. Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv: 1410.3916, 2014.
6. Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. Machine comprehension with syntax, frames, and semantics.
7. Danqi Chen, Jason Bolton, and Christopher D. Manning. A thorough examination of the cnn/daily mail reading comprehension task. In Association for Computational Linguistics (ACL), 2016.
8. Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. Hierarchical question-image co-attention for visual question answering. In https: //araiv. org/abs/ 1603. 01547, 2016.
9. Alessandro Sordoni, Phillip Bachman, and Yoshua Bengio. Iterative alternating neural attention for machine reading. In https: //arziv. org/abs/ 1606. 02245, 2016.
10. Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Hierarchical question-image co-attention for visual question answering. In https: //arziv. org/abs/ 1606. 00061, 2016.
11. Shuohang Wang and Jing Jiang. Machine comprehension using match-lstm and answer pointer. In https: //arziv. org/ abs/ 1608. 07905, 2016.
12. Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems, pp. 2692-2700, 2015.
13. Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In Association for Computational Linguistics (ACL), 2014.
14. Klaus Greff Rupesh Kumar Srivastava and Jurgen Schmidhuber. Highway networks. In https: //araziv. org/abs/ 1505. 00387, 2015.
15. Niki Parmar-Jakob Uszkoreit Llion Jones Aidan N. Gomez Lukasz Kaiser Ashish Vaswani, Noam Shazeer and Illia Polosukhin. Attention is all you need. 2017.
16. Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer open-domain questions. In arXiv preprint arXiv: 1704.00051, 2017.


