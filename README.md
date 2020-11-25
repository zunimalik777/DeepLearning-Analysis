# A Deep Learning Approaches Analysis on Question Classification Task Based on Word2vec Techniques

# Abstract
Question classification is one of the important tasks for automatic question classification in natural language processing (NLP). Recently, there are several text-mining issues such as text classification, documents categorization; web mining, sentiment analysis and spam filtering have been successfully achieved by deep learning approaches. In this study, we illustrated and investigated our study on certain deep learning approaches for question classification tasks in an extremely inflected Turkish language. In this study, we trained and tested the deep learning architectures on questions dataset in Turkish. In addition to this, we used three main deep learning approaches: Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and we also applied two different deep learning combinations of CNN-GRU and CNN-LSTM architectures. Furthermore, we applied Word2vec technique with both skip gram and CBOW methods for word embedding with various vector sizes on a large corpus composed of user questions. By comparing analysis, we conducted experiment of deep learning architectures based on test and 10-cross fold validation accuracy. Experiment results obtained to illustrate that the use of various Word2vec techniques has a considerable impact on the accuracy rate on different deep learning approaches. We attained an accuracy of 93.7% by using these techniques on the question dataset.

# Introduction
Open Turkish Convert (OpenTC Oper Turkish Conversions) is an opensource project for conversions between Traditional Turkish language. It supports character-level and phrase-level conversion, character variant conversion and regional idioms which is translated from UIUC English question dataset.

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
