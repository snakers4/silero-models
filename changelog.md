- [Plans](#plans)
- [Updates](#updates)
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

Current plans w/o any set dates:

- Languages
  - Focus on huge English model update
  - Cover remaning popular Internet languages for CE and EE editions (in progress) - now thinking only about adding French and updating the existing languages
  - We decided not to pursue the following languages: Italian, Portugese, Polish, Czech for now
  - ~~Add some non-core languages only as CE edition~~ Added Ukrainian v1 and v3
- Speed
  - ~~Reduce CE model size to 10-20 MB w/o quality degradation~~ Final size is about 30M w/o losing quality, going lower is still possible with losing quality
  - Add quantized models (x2 speed) - total speed up for smaller models is between 2x and 3x for a `small` models, for `xxsmall` - around 3x
- New products
  - Add denoising - still in progress
  - ~~Add a large number of production grade TTS models - close to finish, soon to be released~~ Added Silero TTS models `v1`

# Updates

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
