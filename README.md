 [![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/silero_speech) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-models/blob/master/LICENSE)

[![Donations](https://opencollective.com/open_stt/tiers/donation/badge.svg?label=donations&color=brightgreen)](https://opencollective.com/open_stt)
[![Backers](https://opencollective.com/open_stt/tiers/backer/badge.svg?label=backers&color=brightgreen)](https://opencollective.com/open_stt)
[![Sponsors](https://opencollective.com/open_stt/tiers/sponsor/badge.svg?label=sponsors&color=brightgreen)](https://opencollective.com/open_stt)

[![Build and Deploy to PyPI](https://github.com/snakers4/silero-models/actions/workflows/build_deploy.yml/badge.svg)](https://github.com/snakers4/silero-models/actions/workflows/build_deploy.yml) [![PyPI version](https://badge.fury.io/py/silero.svg)](https://badge.fury.io/py/silero)

![header](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

- [Silero Models](#silero-models)
  - [Installation and Basics](#installation-and-basics)
  - [Speech-To-Text](#speech-to-text)
    - [Dependencies](#dependencies)
    - [PyTorch](#pytorch)
    - [ONNX](#onnx)
    - [TensorFlow](#tensorflow)
  - [Text-To-Speech](#text-to-speech)
    - [Models and Speakers](#models-and-speakers)
    - [Dependencies](#dependencies-1)
    - [PyTorch](#pytorch-1)
    - [Standalone Use](#standalone-use)
    - [SSML](#SSML)
    - [Indic languages](#indic-languages)
  - [Text-Enhancement](#text-enhancement)
    - [Dependencies](#dependencies-2)
    - [Standalone Use](#standalone-use-1)
  - [FAQ](#faq)
    - [Wiki](#wiki)
    - [Performance and Quality](#performance-and-quality)
    - [Adding new Languages](#adding-new-languages)
  - [Contact](#contact)
    - [Get in Touch](#get-in-touch)
    - [Commercial Inquiries](#commercial-inquiries)
  - [Citations](#citations)
  - [Further reading](#further-reading)
    - [English](#english)
    - [Chinese](#chinese)
    - [Russian](#russian)
  - [Donations](#donations)

# Silero Models

Silero Models: pre-trained enterprise-grade STT / TTS models and benchmarks.

Enterprise-grade STT made refreshingly simple (seriously, see [benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks)).
We provide quality comparable to Google's STT (and sometimes even better) and we are not Google.

As a bonus:

- No Kaldi;
- No compilation;
- No 20-step instructions;

Also we have published TTS models that satisfy the following criteria:

- One-line usage;
- A large library of voices;
- A fully end-to-end pipeline;
- Naturally sounding speech;
- No GPU or training required;
- Minimalism and lack of dependencies;
- Faster than real-time on one CPU thread (!!!);
- Support for 16kHz and 8kHz out of the box;

Also we have published a model for text repunctuation and recapitalization that:

- Inserts capital letters and basic punctuation marks (dot, comma, hyphen, question mark, exclamation mark, dash for Russian);
- Works for 4 languages (Russian, English, German, Spanish) and can be extended;
- By design is domain agnostic and is not based on any hard-coded rules;
- Has non-trivial metrics and succeeds in the task of improving text readability;

## Installation and Basics

You can basically use our models in 3 flavours:

- Via PyTorch Hub: `torch.hub.load()`;
- Via pip:  `pip install silero` and then `import silero`;
- Via caching the required models and utils manually and modifying if necessary;

Models are downloaded on demand both by pip and PyTorch Hub. If you need caching, do it manually or via invoking a necessary model once (it will be downloaded to a cache folder). Please see these [docs](https://pytorch.org/docs/stable/hub.html#loading-models-from-hub) for more information.

PyTorch Hub and pip package are based on the same code. Hence all examples, historically based on `torch.hub.load` can be used with a pip-package via this basic change:

```python3
# before
torch.hub.load(repo_or_dir='snakers4/silero-models',
               model='silero_stt',  # or silero_tts or silero_te
               **kwargs)

# after
from silero import silero_stt, silero_tts, silero_te
silero_stt(**kwargs)
```

## Speech-To-Text

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file.
Any meta-data and newer versions will be added there.

![Screenshot_1](https://user-images.githubusercontent.com/36505480/132320823-f0c5b774-44f7-4375-9c46-3acbcc548b76.png)

Currently we provide the following checkpoints:

|                     | PyTorch            | ONNX               | Quantization       | Quality                                                                         | Colab                                                                                                                                                                    |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| English (`en_v6`)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v6) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| English (`en_v5`)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| German (`de_v4`)    | :heavy_check_mark: | :heavy_check_mark: | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#de-v4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| English (`en_v3`)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| German (`de_v3`)    | :heavy_check_mark: | :hourglass:        | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#de-v3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| German (`de_v1`)    | :heavy_check_mark: | :heavy_check_mark: | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#de-v1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| Spanish (`es_v1`)   | :heavy_check_mark: | :heavy_check_mark: | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#es-v1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| Ukrainian (`ua_v3`) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                                                                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |

Model flavours:

|                   | jit                | jit                | jit                | jit                | jit_q              | jit_q              | onnx               | onnx               | onnx               | onnx               |
| ----------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
|                   | xsmall             | small              | large              | xlarge             | xsmall             | small              | xsmall             | small              | large              | xlarge             |
| English `en_v6`   |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| English `en_v5`   |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| English `en_v4_0` |                    |                    | :heavy_check_mark: |                    |                    |                    |                    |                    | :heavy_check_mark: |                    |
| English `en_v3`   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| German `de_v4`    |                    |                    | :heavy_check_mark: |                    |                    |                    |                    |                    | :heavy_check_mark: |                    |
| German `de_v3`    |                    |                    | :heavy_check_mark: |                    |                    |                    |                    |                    |                    |                    |
| German `de_v1`    |                    | :heavy_check_mark: |                    |                    |                    |                    | :heavy_check_mark: |                    |                    |                    |
| Spanish `es_v1`   |                    | :heavy_check_mark: |                    |                    |                    |                    | :heavy_check_mark: |                    |                    |                    |
| Ukrainian `ua_v3` |                    | :heavy_check_mark: |                    |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    |                    |                    |

### Dependencies

- All examples:
  - `torch`, 1.8+ (used to clone the repo in tf and onnx examples), breaking changes for version older than 1.6
  - `torchaudio`, latest version bound to PyTorch should work
  - `omegaconf`, latest just should work
- Additional for ONNX examples:
  - `onnx`, latest just should work
  - `onnxruntime`, latest just should work
- Additional for TensorFlow examples:
  - `tensorflow`, latest just should work
  - `tensorflow_hub`, latest just should work

Please see the provided Colab for details for each example below. All examples are maintained to work with the latest major packaged versions of the installed libraries.

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_stt/)

```python
import torch
import zipfile
import torchaudio
from glob import glob

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file, any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
test_files = glob('speech_orig.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))
```

### ONNX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

You can run our model everywhere, where you can import the ONNX model or run ONNX runtime.

```python
import onnx
import torch
import onnxruntime
from omegaconf import OmegaConf

language = 'en' # also available 'de', 'es'

# load provided utils
_, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language=language)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

# see available models
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
models = OmegaConf.load('models.yml')
available_languages = list(models.stt_models.keys())
assert language in available_languages

# load the actual ONNX model
torch.hub.download_url_to_file(models.stt_models.en.latest.onnx, 'model.onnx', progress=True)
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('model.onnx')

# download a single file, any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav', dst ='speech_orig.wav', progress=True)
test_files = ['speech_orig.wav']
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]))

# actual onnx inference and decoding
onnx_input = input.detach().cpu().numpy()
ort_inputs = {'input': onnx_input}
ort_outs = ort_session.run(None, ort_inputs)
decoded = decoder(torch.Tensor(ort_outs[0])[0])
print(decoded)
```

### TensorFlow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

**SavedModel example**

```python
import os
import torch
import subprocess
import tensorflow as tf
import tensorflow_hub as tf_hub
from omegaconf import OmegaConf

language = 'en' # also available 'de', 'es'

# load provided utils using torch.hub for brevity
_, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language=language)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

# see available models
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
models = OmegaConf.load('models.yml')
available_languages = list(models.stt_models.keys())
assert language in available_languages

# load the actual tf model
torch.hub.download_url_to_file(models.stt_models.en.latest.tf, 'tf_model.tar.gz')
subprocess.run('rm -rf tf_model && mkdir tf_model && tar xzfv tf_model.tar.gz -C tf_model',  shell=True, check=True)
tf_model = tf.saved_model.load('tf_model')

# download a single file, any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav', dst ='speech_orig.wav', progress=True)
test_files = ['speech_orig.wav']
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]))

# tf inference
res = tf_model.signatures["serving_default"](tf.constant(input.numpy()))['output_0']
print(decoder(torch.Tensor(res.numpy())[0]))
```

## Text-To-Speech

### Models and Speakers

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file. Any meta-data and newer versions will be added there.

#### V3

V3 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

| ID       | Speakers |Auto-stress | Language                           | SR              | Colab                                                                                                                                                                        |
| ------------- | ----------- | ----------- |---------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v3_1_ru`    | `aidar`, `baya`, `kseniya`, `xenia`, `eugene`, `random` | yes  | `ru` (Russian)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_en`    | `en_0`, `en_1`, ..., `en_117`, `random`                   | no   | `en` (English)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_en_indic`   | `tamil_female`, ..., `assamese_male`, `random`       | no   | `en` (English)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_de`    | `eva_k`, ..., `karlsson`, `random`                        | no   | `de` (German)    | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_es`    | `es_0`, `es_1`, `es_2`, `random`                          | no   | `es` (Spanish)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_fr`    | `fr_0`, ..., `fr_5`, `random`                             | no   | `fr` (French)    | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_tt`    | `dilyara`                                                 | no   | `tt` (Tatar)     | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_ua`    | `mykyta`, `random`                                        | no   | `ua` (Ukrainian) | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_uz`    | `dilnavoz`                                                | no   | `uz` (Uzbek)     | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_xal`   | `erdni`, `delghir`, `random`                              | no   | `xal` (Kalmyk)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| [`v3_indic`](#indic-languages)   | `hindi_male`, `hindi_female`, ..., `random`             | no   | `indic` [(Hindi, Telugu, ...)](#indic-languages)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `ru_v3`    | `aidar`, `baya`, `kseniya`, `xenia`, `random`             | yes  | `ru` (Russian)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |


### Dependencies

Basic dependencies for colab examples:

- `torch`, 1.10+;
- `torchaudio`, latest version bound to PyTorch should work (required only because models are hosted together with STT, not required for work);
- `omegaconf`,  latest (can be removed as well, if you do not load all of the configs);

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb)

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_tts/)

```python
# V3
import torch

language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 48000
speaker = 'xenia'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu

audio = model.apply_tts(text=example_text,
                        speaker=speaker,
                        sample_rate=sample_rate)
```

### Standalone Use

- Standalone usage just requires PyTorch 1.10+ and python standard library;
- Please see the detailed examples in Colab;

```python
# V3
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров.'
sample_rate = 48000
speaker='baya'

audio_paths = model.save_wav(text=example_text,
                             speaker=speaker,
                             sample_rate=sample_rate)
```

### SSML

Check out our [TTS Wiki page.](https://github.com/snakers4/silero-models/wiki/SSML)

### Indic languages

#### Example
(!!!) All input sentences should be romanized to ISO format using [`aksharamukha` tool](https://aksharamukha.appspot.com/python). An example for `hindi`:

```python
# V3
import torch
from aksharamukha import transliterate

# Loading model
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v3_indic')

orig_text = "प्रसिद्द कबीर अध्येता, पुरुषोत्तम अग्रवाल का यह शोध आलेख, उस रामानंद की खोज करता है"
roman_text = transliterate.process('Devanagari', 'ISO', orig_text)
print(roman_text)

audio = model.apply_tts(roman_text,
                        speaker='hindi_male')
```

#### Supported languages

| Language | Speakers | Romanization function
-- | -- | --
hindi      | `hindi_female`, `hindi_male`             | `transliterate.process('Devanagari', 'ISO', orig_text)`
malayalam  | `malayalam_female`, `malayalam_male`     |`transliterate.process('Malayalam', 'ISO', orig_text)`
manipuri   | `manipuri_female`                        |`transliterate.process('Bengali', 'ISO', orig_text)`
bengali    | `bengali_female`, `bengali_male`         | `transliterate.process('Bengali', 'ISO', orig_text)`
rajasthani | `rajasthani_female`, `rajasthani_female` | `transliterate.process('Devanagari', 'ISO', orig_text)`
tamil      | `tamil_female`, `tamil_male`             |`transliterate.process('Tamil', 'ISO', orig_text, pre_options=['TamilTranscribe'])`
telugu     | `telugu_female`, `telugu_male`           | `transliterate.process('Telugu', 'ISO', orig_text)`
gujarati   | `gujarati_female`, `gujarati_male`       | `transliterate.process('Gujarati', 'ISO', orig_text)`
kannada    | `kannada_female`, `kannada_male`         |`transliterate.process('Kannada', 'ISO', orig_text)`


## Text-Enhancement

| Languages | Quantization  | Quality | Colab |
| --------- | ------------- | ------- | ----- |
| 'en', 'de', 'ru', 'es' | :heavy_check_mark: | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#te-models) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_te.ipynb) |

### Dependencies

Basic dependencies for colab examples:

- `torch`, 1.9+;
- `pyyaml`, but it's installed with torch itself

### Standalone Use

- Standalone usage just requires PyTorch 1.9+ and python standard library;
- Please see the detailed examples in [Colab](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_te.ipynb);

```python
import torch

model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                  model='silero_te')

input_text = input('Enter input text\n')
apply_te(input_text, lan='en')
```

## FAQ

### Wiki

Also check out our [wiki](https://github.com/snakers4/silero-models/wiki).

### Performance and Quality

Please refer to this wiki sections:

- [Quality Benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks)
- [Performance Benchmarks](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks)

### Adding new Languages

Please refer [here](https://github.com/snakers4/silero-models/wiki/Adding-New-Languages).

## Contact

### Get in Touch

Try our models, create an [issue](https://github.com/snakers4/silero-models/issues/new), join our [chat](https://t.me/silero_speech), [email](mailto:hello@silero.ai) us, read our [news](https://t.me/silero_news).

### Commercial Inquiries

Please see our [wiki](https://github.com/snakers4/silero-models/wiki) and [tiers](https://github.com/snakers4/silero-models/wiki/Licensing-and-Tiers) for relevant information and [email](mailto:hello@silero.ai) us.

## Citations

```bibtex
@misc{Silero Models,
  author = {Silero Team},
  title = {Silero Models: pre-trained enterprise-grade STT / TTS models and benchmarks},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-models}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

## Further reading

### English

- STT:
  - Towards an Imagenet Moment For Speech-To-Text - [link](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/)
  - A Speech-To-Text Practitioners Criticisms of Industry and Academia - [link](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/)
  - Modern Google-level STT Models Released - [link](https://habr.com/ru/post/519562/)

- TTS:
  - Our new public speech synthesis in super-high quality, 10x faster and more stable - [link](https://habr.com/ru/post/660571/) 
  - High-Quality Text-to-Speech Made Accessible, Simple and Fast - [link](https://habr.com/ru/post/549482/)

- VAD:
  - One Voice Detector to Rule Them All - [link](https://thegradient.pub/one-voice-detector-to-rule-them-all/)
  - Modern Portable Voice Activity Detector Released - [link](https://habr.com/ru/post/537276/)

- Text Enhancement:
  - We have published a model for text repunctuation and recapitalization for four languages - [link](https://habr.com/ru/post/581960/) 

### Chinese

- STT:
  - 迈向语音识别领域的 ImageNet 时刻 - [link](https://www.infoq.cn/article/4u58WcFCs0RdpoXev1E2)
  - 语音领域学术界和工业界的七宗罪 - [link](https://www.infoq.cn/article/lEe6GCRjF1CNToVITvNw)

### Russian

- STT
  - Наши сервисы для бесплатного распознавания речи стали лучше и удобнее - [link](https://habr.com/ru/post/654227/) 
  - Telegram-бот Silero бесплатно переводит речь в текст - [link](https://habr.com/ru/post/591563/)
  - Бесплатное распознавание речи для всех желающих - [link](https://habr.com/ru/post/587512/)
  - Последние обновления моделей распознавания речи из Silero Models - [link](https://habr.com/ru/post/577630/) 
  - Сжимаем трансформеры: простые, универсальные и прикладные способы cделать их компактными и быстрыми - [link](https://habr.com/ru/post/563778/)
  - Ультимативное сравнение систем распознавания речи: Ashmanov, Google, Sber, Silero, Tinkoff, Yandex - [link](https://habr.com/ru/post/559640/)
  - Мы опубликовали современные STT модели сравнимые по качеству с Google - [link](https://habr.com/ru/post/519564/)
  - Понижаем барьеры на вход в распознавание речи - [link](https://habr.com/ru/post/494006/)
  - Огромный открытый датасет русской речи версия 1.0 - [link](https://habr.com/ru/post/474462/)
  - Насколько Быстрой Можно Сделать Систему STT? - [link](https://habr.com/ru/post/531524/)
  - Наша система Speech-To-Text - [link](https://www.silero.ai/tag/our-speech-to-text/)
  - Speech To Text - [link](https://www.silero.ai/tag/speech-to-text/)

- TTS:
  - Теперь наш синтез на 20 языках - [link](https://habr.com/ru/post/669910/) 
  - Теперь наш публичный синтез в супер-высоком качестве, в 10 раз быстрее и без детских болячек - [link](https://habr.com/ru/post/660565/) 
  - Синтезируем голос бабушки, дедушки и Ленина + новости нашего публичного синтеза - [link](https://habr.com/ru/post/584750/) 
  - Мы сделали наш публичный синтез речи еще лучше - [link](https://habr.com/ru/post/563484/)
  - Мы Опубликовали Качественный, Простой, Доступный и Быстрый Синтез Речи - [link](https://habr.com/ru/post/549480/)

- VAD:
  - А ты используешь VAD? Что это такое и зачем он нужен - [link](https://habr.com/ru/post/594745/)
  - Модели для Детекции Речи, Чисел и Распознавания Языков - [link](https://www.silero.ai/vad-lang-classifier-number-detector/)
  - Мы опубликовали современный Voice Activity Detector и не только -[link](https://habr.com/ru/post/537274/)

- Text Enhancement:
  - Восстановление знаков пунктуации и заглавных букв — теперь и на длинных текстах - [link](https://habr.com/ru/post/594565/)    
  - Мы опубликовали модель, расставляющую знаки препинания и заглавные буквы в тексте на четырех языках - [link](https://habr.com/ru/post/581946/)

## Donations

Please use the "sponsor" button.
