# A Deep Learning Approaches Analysis on Question Classification Task Based on Word2vec Techniques

# Abstract
Question classification is one of the important tasks for automatic question classification in natural language processing (NLP). Recently, there are several text-mining issues such as text classification, documents categorization; web mining, sentiment analysis and spam filtering have been successfully achieved by deep learning approaches. In this study, we illustrated and investigated our study on certain deep learning approaches for question classification tasks in an extremely inflected Turkish language. In this study, we trained and tested the deep learning architectures on questions dataset in Turkish. In addition to this, we used three main deep learning approaches: Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and we also applied two different deep learning combinations of CNN-GRU and CNN-LSTM architectures. Furthermore, we applied Word2vec technique with both skip gram and CBOW methods for word embedding with various vector sizes on a large corpus composed of user questions. By comparing analysis, we conducted experiment of deep learning architectures based on test and 10-cross fold validation accuracy. Experiment results obtained to illustrate that the use of various Word2vec techniques has a considerable impact on the accuracy rate on different deep learning approaches. We attained an accuracy of 93.7% by using these techniques on the question dataset.

# Introduction
This tutorial introduces how to train word2vec model for Turkish language from Wikipedia dump. This code is written in Python 3 by using gensim(https://radimrehurek.com/gensim/) library. Turkish is an agglutinative language and there are many words with the same lemma and different suffixes in the wikipedia corpus. I will write Turkish lemmatizer to increase quality of the model. You can checkout wiki-page for more details. If you can look for examples in 5. Using Word2Vec Model and Examples page in github wiki.

# Getting the Corpus
We need to have big corpus to train word2vec model. You can access all wikipedia articles written in Turkish language from wikimedia dumps (https://dumps.wikimedia.org/trwiki/). The available one is 20180101 for this day and you can download all articles until 01/01/2018 by this link, 20180101. Of course, you can use another corpus to train word2vec model but you must modify your corpus to train a model with gensim library, explained following link below
https://www.python.org/download/releases/3.0/

# Preprocessing the Corpus
To train word2vec model with gensim library, you need to put each document into a line without punctuations. So, the output file should include all articles and each article should be in a line. Gensim library provides methods to do this preprocessing step. However, tokenize function is modified for Turkish language. You can run preprocess.py to modify your wikipedia dump corpus. It takes two arguments. First one is the path to the wikipedia dump(without extracting). Second one is the path to the output file. For example:

python3 preprocess.py trwiki-20180101-pages-articles.xml.bz2 wiki.tr.txt

# Training Word2Vec Model
After preprocessing the corpus, training word2vec model with gensim library is very easy. You can use the code below to create word2vec model. First argument is revised corpus and the second one is name of the model.

python3 trainCorpus.py wiki.tr.txt trmodel

This command creates an output file which is binary file for vectors for words. You can download pretrained model (https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww) which is created by following steps in this tutorial.

# JavaScript
const OpenTC = require('opencc');
const converter = new OpenTC('uiuc.word');
converter.convertPromise("Turkish character").then(converted => {
  console.log(converted);  // Turkish character
});
# TypeScript
import { OpenTC } from 'opentc';
async function main() {
  const converter: OpenTC = new OpenTC('uiuc.word');
  const result: string = await converter.convertPromise('Turkish character');
  console.log(result);
}

# Python
PyPI pip install opentc (Windows, Linux, Mac)
import opentc
converter = opentc.OpenTC('uiuc.word')
converter.convert('Turkish')  #Turkish character

# C++
#include "opencc.h"

int main() {
  const SimpleConverter converter("uiuc.word");
  converter.Convert("Turkish character");  // Turkish character
  return 0;
}
# C
#include "opentc.h"

int main() {
  opentc_t opentc = opentc_open("uiuc.word");
  const char* input = "Turkish character";
  char* converted = opentc_convert_utf8(opentc, input, strlen(input));  // Turkish character
  opencc_convert_utf8_free(converted);
  opencc_close(opentc);
  return 0;
}

# Command Line
opencc --help
opencc_dict --help
opencc_phrase_extract --help

# Others (Unofficial)
Swift (iOS): SwiftyOpenTC
Java: opentc4j
Android: android-opentc
PHP: opentc4php
Pure JavaScript: opentc-js
WebAssembly: wasm-opentc
