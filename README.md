 [![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/silero_speech) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-GNU%20AGPL%203.0-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-models/blob/master/LICENSE)

[![Donations](https://opencollective.com/open_stt/tiers/donation/badge.svg?label=donations&color=brightgreen)](https://opencollective.com/open_stt)
[![Backers](https://opencollective.com/open_stt/tiers/backer/badge.svg?label=backers&color=brightgreen)](https://opencollective.com/open_stt)
[![Sponsors](https://opencollective.com/open_stt/tiers/sponsor/badge.svg?label=sponsors&color=brightgreen)](https://opencollective.com/open_stt)

![header)](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

- [Silero Models](#silero-models)
  - [Speech-To-Text](#speech-to-text)
    - [Dependencies](#dependencies)
    - [PyTorch](#pytorch)
    - [ONNX](#onnx)
    - [TensorFlow](#tensorflow)
  - [Text-To-Speech](#text-to-speech)
    - [Dependencies](#dependencies-1)
    - [PyTorch](#pytorch-1)
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

## Speech-To-Text

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file.
Any meta-data and newer versions will be added there.

Currently we provide the following checkpoints:

|                     | PyTorch            | ONNX               | Quantization       | Quality                                                                         | Colab                                                                                                                                                                    |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| English (`en_v3`)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#en-v3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| German (`de_v1`)    | :heavy_check_mark: | :heavy_check_mark: | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#de-v1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| Spanish (`es_v1`)   | :heavy_check_mark: | :heavy_check_mark: | :hourglass:        | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#es-v1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| Ukrainian (`ua_v3`) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                                                                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |

Model flavours:

|                    | jit                | jit                | jit                | jit_q              | jit_q              | onnx               | onnx               | onnx               |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
|                    | xsmall             | small              | large              | xsmall             | small              | xsmall             | small              | large              |
| English (`en_v3`)  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| German (`de_v1`)   |                    | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: |                    |                    |
| Spanish (`es_v1`)  |                    | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: |                    |                    |
| Ukrainian (`ua_v3` |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: |                    |                    |

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

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file. Any meta-data and newer versions will be added there.

Currently we provide the following speakers:

| Speaker          | Stress | Language | SR    | PyTorch            | Colab                                                                                                                                                                        |
| ---------------- | ------ | -------- | ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aidar_8khz`     | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `baya_8khz`      | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `ksenia_8khz`    | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `irina_8khz`     | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `natasha_8khz`   | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `ruslan_8khz`    | yes    | `ru`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `lj_8khz`        | no     | `en`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `thorsten_8khz`  | no     | `de`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `gilles_8khz`    | no     | `fr`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `tux_8khz`       | no     | `es`     | 8000  | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `aidar_16khz`    | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `baya_16khz`     | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `ksenia_16khz`   | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `irina_16khz`    | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `natasha_16khz`  | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `ruslan_16khz`   | yes    | `ru`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `lj_16khz`       | no     | `en`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `thorsten_16khz` | no     | `de`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `gilles_16khz`   | no     | `fr`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `tux_16khz`      | no     | `es`     | 16000 | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |

### Dependencies

Basic dependencies (see colab):

- `torch`, 1.8+ (used to clone the repo in tf and onnx examples), breaking changes for version older than 1.6
- `torchaudio`, latest version bound to PyTorch should work (required only because models are hosted together with STT, not required for work)
- `omegaconf`,  latest (can be removed as well, if you do not load all of the configs)

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb)

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_tts/)

```python
import torch

language = 'ru'
speaker = 'kseniya_16khz'
device = torch.device('cpu')
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
model = model.to(device)  # gpu or cpu
audio = apply_tts(texts=[example_text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)
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
  - High-Quality Text-to-Speech Made Accessible, Simple and Fast - [link](https://habr.com/ru/post/549482/)

- VAD:
  - Modern Portable Voice Activity Detector Released - [link](https://habr.com/ru/post/537276/)

### Chinese

- STT:
  - 迈向语音识别领域的 ImageNet 时刻 - [link](https://www.infoq.cn/article/4u58WcFCs0RdpoXev1E2)
  - 语音领域学术界和工业界的七宗罪 - [link](https://www.infoq.cn/article/lEe6GCRjF1CNToVITvNw)

### Russian

- STT
  - Мы опубликовали современные STT модели сравнимые по качеству с Google - [link](https://habr.com/ru/post/519564/)
  - Понижаем барьеры на вход в распознавание речи - [link](https://habr.com/ru/post/494006/)
  - Огромный открытый датасет русской речи версия 1.0 - [link](https://habr.com/ru/post/474462/)
  - Насколько Быстрой Можно Сделать Систему STT? - [link](https://habr.com/ru/post/531524/)
  - Наша система Speech-To-Text - [link](https://www.silero.ai/tag/our-speech-to-text/)
  - Speech To Text - [link](https://www.silero.ai/tag/speech-to-text/ )

- TTS:
  - Мы Опубликовали Качественный, Простой, Доступный и Быстрый Синтез Речи - [link](https://habr.com/ru/post/549480/)

- VAD:
  - Мы опубликовали современный Voice Activity Detector и не только -[link](https://habr.com/ru/post/537274/)

## Donations

Please use the "sponsor" button.
