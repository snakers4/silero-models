- [Plans](#plans)
- [Updates](#updates)
  - [2020-09-29 Published on TF Hub](#2020-09-29-published-on-tf-hub)
  - [2020-09-28 Added Timestamps To Decoder](#2020-09-28-added-timestamps-to-decoder)
  - [2020-09-27 Examples, Usability, TF Example](#2020-09-27-examples-usability-tf-example)
  - [2020-09-23 Fixed broken TF Model Archives](#2020-09-23-fixed-broken-tf-model-archives)
  - [2020-09-23 Silero Models now on Torch Hub](#2020-09-23-silero-models-now-on-torch-hub)
  - [2020-09-22 Tensorflow SavedModels, Colab VAD, tf.js](#2020-09-22-tensorflow-savedmodels-colab-vad-tfjs)
  - [2020-09-19 Fx Minor Bugs](#2020-09-19-fx-minor-bugs)
  - [2020-09-17 TorchHub](#2020-09-17-torchhub)
  - [2020-09-16 V1 Release](#2020-09-16-v1-release)
  - [2020-09-15 Quality Benchmarks for German](#2020-09-15-quality-benchmarks-for-german)
  - [2020-09-12 Quality Benchmarks for English](#2020-09-12-quality-benchmarks-for-english)
  - [2020-09-11 Initial upload](#2020-09-11-initial-upload)

# Plans

General plans w/o any set dates:

- Cover remaning popular Internet languages for CE and EE editions
- Reduce CE model size to 10-20 MB w/o quality degradation
- Add some non-core languages only as CE edition
- Add denoising
- Add quantized models (x2 speed)

# Updates

## 2020-09-29 Update English benchmarks

new pruned double-lm [quality benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v1)

## 2020-09-29 Published on TF Hub 

- https://tfhub.dev/silero
- https://tfhub.dev/silero/silero-stt/en/1
- https://tfhub.dev/silero/collections/silero-stt/1

## 2020-09-28 Added Timestamps To Decoder

- Now standard decoder has an option to return word timestamps
- Please see Colab examples for PyTorch

## 2020-09-27 Examples, Usability, TF Example 

- Polish and simplify the main readme
- Remove folders from inside of TF archives
- Polish model url naming, purge CDN cache
- Add TF Colab example
- Remove old ONNX example
- Submit to TF and ONNX hub

## 2020-09-23 Fixed broken TF Model Archives 

- Fixed weird archiving issue for Windows, purged CDN cache

## 2020-09-23 Silero Models now on Torch Hub

- https://pytorch.org/hub/snakers4_silero-models_stt/

## 2020-09-22 Tensorflow SavedModels, Colab VAD, tf.js

- Add VAD to colab example
- Add proper SavedModel Tensorflow checkpoints for all of the languages
- Add experimental tf.js checkpoints for English 

## 2020-09-19 Fx Minor Bugs

Fix minor colab bugs

## 2020-09-17 TorchHub

- Added loading via TorchHub
- Added issue templates
- Fix typos

## 2020-09-16 V1 Release

- Examples and docs further polished
- Added a more thorough colab example with file upload / speech recording
- Added performance benchmarks
- Added quality benchmarks for Spanish in the wiki

## 2020-09-15 Quality Benchmarks for German

Added quality benchmarks for German in the wiki

## 2020-09-12 Quality Benchmarks for English

Added quality benchmarks for English in the wiki

## 2020-09-11 Initial upload

First commit and first models uploaded:

- English
- German
- Spanish
