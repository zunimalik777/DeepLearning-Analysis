# A Deep Learning Approaches Analysis on Question Classification Task Based on Word2vec Techniques

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
