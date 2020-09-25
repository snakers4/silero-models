 [![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/joinchat/Bv9tjhpdXTI22OUgpOIIDg)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-GNU%20AGPL%203.0-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-models/blob/master/LICENSE) [![License: CC BY-NC 4.0](https://img.shields.io/badge/Torch-Open%20in%20Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_stt/)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

![header)](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)


- [Silero Models](#silero-models)
  - [Getting Started](#getting-started)
    - [PyTorch](#pytorch)
    - [ONNX](#onnx)
    - [TensorFlow](#tensorflow)
  - [Wiki](#wiki)
  - [Performance and Quality](#performance-and-quality)
  - [Adding new Languages](#adding-new-languages)  
  - [Get in Touch](#get-in-touch)
  - [Commercial Inquiries](#commercial-inquiries)


# Silero Models

Silero Models: pre-trained enterprise-grade STT models and benchmarks.
Enterprise-grade STT made refreshingly simple (seriously, see [benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks)).
We provide quality comparable to Google's STT (and sometimes even better) and we are not Google.

As a bonus:

- No Kaldi.
- No compilation.
- No 20-step instructions.

## Getting Started

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file.
Any meta-data and newer versions will be added there.

Currently we provide the following checkpoints:

|                 | PyTorch            | ONNX               | TensorFlow         | Quantization | Quality | Colab | 
|-----------------|--------------------|--------------------|--------------------|--------------|---------|-------| 
| English (en_v1) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :hourglass:  | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#latest) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| German (de_v1)  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :hourglass:  | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#latest) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |
| Spanish (es_v1) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :hourglass:  | [link](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks#latest) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb) |

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

**Dependencies:**

- PyTorch 1.6+
- TorchAudio 0.7+ (you can use your own data loaders)
- omegaconf (or any similar library to work with yaml files)


**Loading a model is as easy as cloning this repository and:**

```python
import torch
from omegaconf import OmegaConf

models = OmegaConf.load('models.yml')
device = torch.device('cpu')   # you can use any pytorch device
model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)
```

**Or you can just use TorchHub:**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/Torch-Open%20in%20Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_stt/)

`torch.hub` clones the repo for you behind the scenes.

```python
import torch

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(github='snakers4/silero-models',
                                       model='silero_stt',
                                       device=device,
                                       force_reload=True,
                                       language='de')

(read_batch,
 split_into_batches,
 read_audio,
 prepare_model_input) = utils  # see function signatures for details
```

We provide our models as TorchScript packages, so you can use the deployment options PyTorch itself provides (C++, Java). See details in the [example](https://github.com/snakers4/silero-models/blob/master/examples.ipynb) notebook.


### ONNX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb)

You can run our model everywhere, where you can import the ONNX model or run ONNX runtime.

**Dependencies:**

- PyTorch 1.6+ (used for utilities only)
- omegaconf (or any similar library to work with yaml files)
- onnx
- onnxruntime

**Just clone the repo and**:

```python
import json
import onnx
import torch
import tempfile
import onnxruntime
from omegaconf import OmegaConf

models = OmegaConf.load('models.yml')

with tempfile.NamedTemporaryFile('wb', suffix='.json') as f:
    torch.hub.download_url_to_file(models.stt_models.en.latest.labels,
                                   f.name,
                                   progress=True)
    with open(f.name) as f:
        labels = json.load(f)
        decoder = Decoder(labels)

with tempfile.NamedTemporaryFile('wb', suffix='.model') as f:
    torch.hub.download_url_to_file(models.stt_models.en.latest.onnx,
                                   f.name,
                                   progress=True)
    onnx_model = onnx.load(f.name)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(f.name)
```

See details in the [example](https://github.com/snakers4/silero-models/blob/master/examples.ipynb) notebook.

### TensorFlow

(Experimental) We provide several types of Tensorflow checkpoints:

- Tensorflow SavedModel
- `tf-js` float32 model
- `tf-js` int8 model

Loading a SavedModel:

```python
import os
import torch
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # maybe there is a more proper way for TF
tf_model = tf.saved_model.load("tf_models/tf_en_v1/saved_model")

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(github='snakers4/silero-models',
                                       model='silero_stt',
                                       device=device,
                                       force_reload=True,
                                       language='en')
(read_batch,
 split_into_batches,
 read_audio,
 prepare_model_input) = utils

wav_paths = random.sample(en_val_files, k=1)
_batch = [read_audio(wav_path) for wav_path in wav_paths]
batches = split_into_batches(_batch, batch_size=1)

# PyTorch inference
inputs = get_model_input(random.sample(batches, k=1)[0])
out_sm = model(inputs)
decoded = decoder(out_sm[0])
print(decoded)

# TF inference
res = tf_model.signatures["serving_default"](tf.constant(inputs.numpy()[0]))['output_0']
print(decoder(torch.Tensor(res.numpy())))

```

## Wiki

Also check out our [wiki](https://github.com/snakers4/silero-models/wiki).

## Performance and Quality

Please refer to this wiki sections:

- [Quality Benchmarks](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks)
- [Performance Benchmarks](https://github.com/snakers4/silero-models/wiki/Performance-Benchmarks)

## Adding new Languages

Please refer [here](https://github.com/snakers4/silero-models/wiki/Adding-New-Languages).

## Get in Touch

Try our models, create an [issue](https://github.com/snakers4/silero-models/issues/new), join our [chat](https://t.me/joinchat/Bv9tjhpdXTI22OUgpOIIDg), [email](mailto:hello@silero.ai) us.

## Commercial Inquiries

Please see our [wiki](https://github.com/snakers4/silero-models/wiki) and [tiers](https://github.com/snakers4/silero-models/wiki/Licensing-and-Tiers) for relevant information and [email](mailto:hello@silero.ai) us.
