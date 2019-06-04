# NLP_Tensorflow2.0

This is NLP_with_MentoringProgram with DeepNLP

Most of information is from [NLP paper implementation with PyTorch](https://github.com/aisolab/nlp_implementation) 

### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected.

|                  | Train ACC (120,000) | Validation ACC (30,000) | Test ACC (50,000) |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)     |  92.33%  |   -   | 81.29%  |
| SenCNN           |  92.22%  |     86.81%     |  86.48%  |
| SenCNN(Ryan)     |  92.53%  |     -     |  82.99%  |
| SenCNN(SM)       |  92.3%   |     84.98%     |  84.42%  |
| SenCNN(JMKIM)    |  94.56%  |     86.268%     |  85.851%  |
| CharCNN          | - | - | - |
| ConvRec          | - | - | - |
| VDCNN            | - | - | - |
| SAN | - | - | - |

* [ ] [Convolutional Neural Networks for Sentence Classification](https://github.com/aisolab/nlp_implementation/tree/master/Convolutional_Neural_Networks_for_Sentence_Classification) (SenCNN)
  + https://arxiv.org/abs/1408.5882
* [ ] [Character-level Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Character-level_Convolutional_Networks_for_Text_Classification) (CharCNN)
  + https://arxiv.org/abs/1509.01626
* [ ] [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://github.com/aisolab/nlp_implementation/tree/master/Efficient_Character-level_Document_Classification_by_Combining_Convolution_and_Recurrent_Layers) (as ConvRec)
  + https://arxiv.org/abs/1602.00367
* [ ] [Very Deep Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Very_Deep_Convolutional_Networks_for_Text_Classification) (as VDCNN)
  + https://arxiv.org/abs/1606.01781
* [ ] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding) (as SAN)
  + https://arxiv.org/abs/1703.03130

### Sentence Simlarity
+ Using the [Question_pair from songys](https://github.com/songys/Question_pair)
+ Hyper-parameter was arbitrarily selected.
+ Most of approaches are in [SNLI from Stanford](https://nlp.stanford.edu/projects/snli/)

* [ ] Learning Sentence Similarity with Siamese Recurrent Architectures
	+ https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023
* [ ] Fine-Tuned LM-Pretrained Transformer
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf



### Neural machine translation
* [ ] Effective Approaches to Attention-based Neural Machine Translation
	+ https://arxiv.org/abs/1508.04025
* [ ] Attention Is All You Need
	+ https://arxiv.org/abs/1706.03762

### Machine reading comprension
* [ ] Bi-directional attention flow for machine comprehension
	+ https://arxiv.org/abs/1611.01603

### Transfer learning
* [ ] Deep contextualized word representations
	+ https://arxiv.org/abs/1802.05365
* [ ] Improving Language Understanding by Generative Pre-Training
	+ https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
* [ ] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
	+ https://arxiv.org/abs/1810.04805
* [ ] Language Models are Unsupervised Multitask Learners
	+ https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

### Named entity recognition
+ Using the [Naver nlp-challange corpus for NER](https://github.com/naver/nlp-challenge/tree/master/missions/ner)
+ Hyper-parameter was arbitrarily selected.
* [ ] Bidirectional LSTM-CRF Models for Sequence Tagging
	+ https://arxiv.org/abs/1508.01991
* [ ] End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
	+ https://arxiv.org/abs/1603.01354
* [ ] Neural Architectures for Named Entity Recognition
	+ https://arxiv.org/abs/1603.01360

### Language model
* [ ] Character-Aware Neural Language Models
  + https://arxiv.org/abs/1508.06615

