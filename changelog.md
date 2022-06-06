- [Plans](#plans)
- [Updates](#updates)
  - [2022-06-06 Silero TTS in 20 Languages With 174 Speakers](#2022-06-06-silero-tts-in-20-languages-with-174-speakers)
  - [2022-04-12 Silero TTS in High Resolution, 10x Faster and More Stable](#2022-04-12-silero-tts-in-high-resolution-10x-faster-and-more-stable)
  - [2022-02-28 Experimental Pip Package](#2022-02-28-experimental-pip-package)
  - [2022-02-24 English V6 Release](#2022-02-24-english-v6-release)
  - [2021-12-09 Improved Text Recapitalization and Repunctuation Model for 4 Languages](#2021-12-09-improved-text-recapitalization-and-repunctuation-model-for-4-languages)
  - [2021-10-06 Text Recapitalization and Repunctuation Model for 4 Languages](#2021-10-06-text-recapitalization-and-repunctuation-model-for-4-languages)
  - [2021-09-03 German V4 and English V5 Models](#2021-09-03-german-v4-and-english-v5-models)
  - [2021-08-09 German V3 Large Model](#2021-08-09-german-v3-large-model)
  - [2021-06-18 Large V2 TTS release, v4_0 Large English STT Model](#2021-06-18-large-v2-tts-release-v4_0-large-english-stt-model)
  - [2021-04-21 Large V2 TTS release, v4_0 Large English STT Model](#2021-04-21-large-v2-tts-release-v4_0-large-english-stt-model)
  - [2021-04-20 Add `v3` STT English Models](#2021-04-20-add-v3-stt-english-models)
  - [2021-03-29 Add `v1` TTS Models](#2021-03-29-add-v1-tts-models)
  - [2021-03-03 Add `xxsmall` Speed Metrics](#2021-03-03-add-xxsmall-speed-metrics)
  - [2021-03-03 Ukrainian Model V3 Released](#2021-03-03-ukrainian-model-v3-released)
  - [2021-02-15 Some Organizational Issues](#2021-02-15-some-organizational-issues)
  - [2020-12-04 Add EE Distro Sizing and New Speed Metrics](#2020-12-04-add-ee-distro-sizing-and-new-speed-metrics)
  - [2020-11-26 Fix TensorFlow Examples](#2020-11-26-fix-tensorflow-examples)
  - [2020-11-03 [Experimental] Ukrainian Model V1 Released](#2020-11-03-experimental-ukrainian-model-v1-released)
  - [2020-11-03 English Model V2 Released](#2020-11-03-english-model-v2-released)
  - [2020-10-28 Minor PyTorch 1.7 fix](#2020-10-28-minor-pytorch-17-fix)
  - [2020-10-19 Update wiki](#2020-10-19-update-wiki)
  - [2020-10-03 Batched ONNX and TF Models](#2020-10-03-batched-onnx-and-tf-models)
  - [2020-09-29 Update English benchmarks](#2020-09-29-update-english-benchmarks)
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

# Updates

## 2022-06-06 Silero TTS in 20 Languages With 174 Speakers

- Huge [release](https://habr.com/ru/post/669910/) - 20 languages, 173 voices;
- 1 new high quality Russian voice (`eugeny`);
- The CIS languages: Kalmyk, Russian, Tatar, Uzbek и Ukrainian;
- Romance and Germanic languages: English, Indic English, Spanish, German, French;
- 10 Indic languages;
- Russian automated stress model vastly improved (please see this [link](https://habr.com/ru/post/669910/) for details);
- All models inherit all of the previous SSML perks;

## 2022-04-12 Silero TTS in High Resolution, 10x Faster and More Stable

- Huge [release](https://habr.com/ru/post/660571/) - Russian only for now;
- Model size reduced 2x;
- New models are 10x faster;
- We added flags to control stress;
- Now the models can make proper pauses;
- High quality voice added (and unlimited "random" voices);
- All speakers squeezed into the same model;
- Input length limitations lifted, now models can work with paragraphs of text;
- Pauses, speed and pitch can be controlled via SSML;
- Sampling rates of 8, 24 or 48 kHz are supported;
- Models are much more stable — they do not omit words anymore;

## 2022-02-28 Experimental Pip Package

- Models are downloaded on demand both by `pip` and PyTorch Hub;
- If you need caching, do it manually or via invoking a necessary model once (it will be downloaded to a cache folder);
- Please see these docs for more information;
- PyTorch Hub and pip package are based on the same code. Hence all examples, historically based on torch.hub.load can be used with a pip-package;

## 2022-02-24 English V6 Release

- New `en_v6` models;
- Quality improvements for English models;

## 2021-12-09 Improved Text Recapitalization and Repunctuation Model for 4 Languages

- The model now can work with long inputs, 512 tokens or ca. 150 words;
- Inputs longer than 150 words are automatically processed in chunks;
- The bugs with newer PyTorch versions have been fixed;
- Model was trained longer with larger batches;
- Model size slightly reduced to 85 MB;
- The rest of model optimizations were deemed too high maintenance;

## 2021-10-06 Text Recapitalization and Repunctuation Model for 4 Languages

- Inserts capital letters and basic punctuation marks (dot, comma, hyphen, question mark, exclamation mark, dash for Russian);
- Works for 4 languages (Russian, English, German, Spanish) and can be extended;
- By design is domain agnostic and is not based on any hard-coded rules;
- Has non-trivial metrics and succeeds in the task of improving text readability;

## 2021-09-03 German V4 and English V5 Models

- German V4 large `jit` and `onnx` models;
- English V5 `small` (`jit` and `onnx`), `small_q` (only `jit`) and `xlarge`  (`jit` and `onnx`) models;
- Vast quality improvements (metrics to be added shortly) on the majority of domains;
- English `xsmall` models coming soon (`jit` and `onnx`);

## 2021-08-09 German V3 Large Model

- German V3 Large `jit` model trained on more data - large quality improvement;
- Metrics coming soon;

## 2021-06-18 Large V2 TTS release, v4_0 Large English STT Model

- Added v4_0 large English model with metrics;
- V2 TTS models with x4 faster vocoder;
- Russian models now feature automatic stress and `ё`, homonyms are not handled yet;
- A multi-language multi-speaker model;

## 2021-04-21 Large V2 TTS release, v4_0 Large English STT Model

Huge update for English!

- Polish docs;
- Add `xsmall` and `xsmall_q` model flavours for `en_v3`;
- Polish performance benchmarks page a bit;

## 2021-04-20 Add `v3` STT English Models

Huge update for English!

- Default model (`jit` or `onnx`) size is reduced almost by 50% without sacrificing quality (!);
- New model flavours: `jit_q` (smaller quantized model), `jit_skip` (with exposed skip connections), `jit_large` (higher quality model), `onnx_large` (!);
- New smallest model `jit_q` is only 40M in size (!);
- Tensorflow checkpoints discontinued;
- New performance [benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v3) - default models are on par with previous models and Google, large models mostly outperform Google (!);
- Even more quality improvements coming soon (!);
- CE benchmarks coming soon;
- `xsmall` model was created (2x smaller than the default), but I could not quantize it. I am looking into creating a `xxsmall` model;
- Still working on making EE models fully JIT-traceable;

## 2021-03-29 Add `v1` TTS Models

- Added `v1` TTS [models](https://github.com/snakers4/silero-models#text-to-speech);
- Add TTS [performance benchmarks](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#tts);
- Polish existing [wiki](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks);
- Progress on an additional method of [model compression](https://t.me/snakers4/2689);

## 2021-03-03 Add `xxsmall` Speed Metrics

- See metrics [here](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#speed-benchmarks)
- Note that this is only an acoustic model, full end-to-end system metrics differ, though `xxsmall` metrics trickle down for CPU systems

## 2021-03-03 Ukrainian Model V3 Released

- Fine tuned from a commercial production Russian model
- Trained on a larger corpus (around 1,000 hours)
- Model flavors: `jit` (CPU or GPU), `jit_q` (quantized and CPU only) and `onnx` (ONNX)
- Huge model speed improvements for CPU inference (!roughly 3x faster!) compared to the previous one, comparable with `xxsmall` from [here](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks#speed-benchmarks)
- Will be dropping TF support altogether
- No proper quality benchmarks for an experimental model though

## 2021-02-15 Some Organizational Issues

- Migrate to our own model hosting
- Solve large parasite traffic / DDOS issue, source still unknown
- Remove the CDN
- Some community / answers tidying up
- Major progress on [silero-vad](https://github.com/snakers4/silero-vad)
- Major progress on TTS, preparing for a release

## 2020-12-04 Add EE Distro Sizing and New Speed Metrics

- https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks

## 2020-11-26 Fix TensorFlow Examples

## 2020-11-03 [Experimental] Ukrainian Model V1 Released

- An experimental model
- Trained from a small community contributed [corpus](https://github.com/snakers4/silero-models/issues/30)
- **New** Full model size reduced to 85 MB
- **New** - quantized model is ony 25 MB
- No TF or ONNX models
- Will be re-released a fine-tuned model from a larger Russian corpus upon V3 release

## 2020-11-03 English Model V2 Released

- A minor release, i.e. other models not affected
- English model was made much more robust to certain dialects
- Performance metrics coming soon

## 2020-10-28 Minor PyTorch 1.7 fix

- torch.hub.load signature was changed

## 2020-10-19 Update wiki

- Add article on [Methodology](https://github.com/snakers4/silero-models/wiki/Methodology), update wiki

## 2020-10-03 Batched ONNX and TF Models

- Extensively clean up and simplify ONNX and TF model code
- Add batch support to TF and ONNX models
- Update examples
- (pending) Submit new models to TF Hub and update examples there

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
