 [![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/silero_speech) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-models/blob/master/LICENSE)

[![PyPI version](https://badge.fury.io/py/silero.svg)](https://badge.fury.io/py/silero)

![header](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

- [Silero Models](#silero-models)
  - [Installation and Basics](#installation-and-basics)
  - [Text-To-Speech](#text-to-speech)
    - [Models and Speakers](#models-and-speakers)
      - [V5](#v5)
      - [V5 CIS Base Models](#v5-cis-base-models)
      - [V5 CIS Ext Models](#v5-cis-ext-models)
      - [V4](#v4)
      - [V3](#v3)
    - [Dependencies](#dependencies)
    - [PyTorch](#pytorch)
    - [Standalone Use](#standalone-use)
    - [SSML](#ssml)
    - [Cyrillic languages v4](#cyrillic-languages-v4)
    - [Indic languages v4](#indic-languages-v4)
      - [Example](#example)
      - [Supported languages](#supported-languages)
  - [Contact](#contact)
  - [Licence](#licence) 
  - [Citations](#citations)
  - [Further reading](#further-reading)
    - [English](#english)
    - [Chinese](#chinese)
    - [Russian](#russian)

# Silero Models

Our TTS models satisfy the following criteria:

- Fully end-to-end;
- Large library of voices;
- Natural-sounding speech;
- One-line usage, minimal, portable;
- Impressively fast on CPU and GPU;
- For the Russian language - automated stress and homographs;
 
## Installation and Basics

You can basically use our models in 3 flavours:

- Via PyTorch Hub: `torch.hub.load()`;
- Via pip:  `pip install silero` and then `from silero import silero_tts`;
- Via caching the required models and utils manually and modifying if necessary;

Models are downloaded on demand both by pip and PyTorch Hub. If you need caching, do it manually or via invoking a necessary model once (it will be downloaded to a cache folder). Please see these [docs](https://pytorch.org/docs/stable/hub.html#loading-models-from-hub) for more information.

PyTorch Hub and pip package are based on the same code. All of the `torch.hub.load` examples can be used with the pip package via this basic change:

```python
from silero import silero_tts
model, example_text = silero_tts(language='ru',
                                 speaker='v5_ru')
audio = model.apply_tts(text=example_text)
```

## Text-To-Speech

### Models and Speakers

All of the provided models are listed in the [models.yml](https://github.com/snakers4/silero-models/blob/master/models.yml) file. Any metadata and newer versions will be added there.

#### V5

V5 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

Russian-only models support automated stress and homographs.

| ID      | Speakers                                      | Auto-stress / Homographs | Language       | SR                       | Colab                                                                                                                                                                        |
| ------- | --------------------------------------------- | ----------- | -------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v5_ru` | `aidar`, `baya`, `kseniya`, `xenia`, `eugene` | yes / yes        | `ru` (Russian) | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |

#### V5 CIS Base Models

- All of the below models support `8000`, `24000`, `48000` sampling rates and contain no auto-stress or homographs;
- `v5_cis_base` models assume that proper stress should be added for each word for all languages, i.e. `к+ошка`;
- `v5_cis_base_nostress` models assume that proper stress should be added for each word ONLY for slavic languages (i.e. `ru`, `bel`, `ukr`); 
- All of the below models are published under `MIT` licence;
- V5 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

| ID                                    | Speakers                                     | Language             | Colab |
| ------------------------------------- | -------------------------------------------- | -------------------- | -------------------- |
| `v5_cis_base`, `v5_cis_base_nostress` | `aze_gamat`                                  | `aze` (Azerbaijani)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `hye_zara`                                   | `hye` (Armenian)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `bak_aigul`, `bak_alfia`, `bak_alfia2`       | `bak` (Bashkir)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `bak_miyau`, `bak_ramilia`                   | `bak` (Bashkir)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `bel_anatoliy`, `bel_dmitriy`, `bel_larisa`  | `bel` (Belarus)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `kat_vika`                                   | `kat` (Georgian)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `kbd_eduard`                                 | `kbd` (Kab.-Cherkes) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `kaz_zhadyra`, `kaz_zhazira`                 | `kaz` (Kazakh)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `xal_kejilgan`, `xal_kermen`                 | `xal` (Kalmyk)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `kir_nurgul`                                 | `kir` (Kyrgyz)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `mdf_oksana`                                 | `mdf` (Moksha)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | all of these speakers, but with `ru_` prefix | `ru`  (Russian)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `tgk_onaoy`, `tgk_safarhuja`                 | `tgk` (Tajik)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `tat_albina`, `tat_marat`                    | `tat` (Tatar)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `udm_bogdan`                                 | `udm` (Udmurt)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `uzb_saida`                                  | `uzb` (Uzbek)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `ukr_igor`, `ukr_roman`                      | `ukr` (Ukrainian)    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `kjh_karina`, `kjh_sibday`                   | `kjh` (Khakas)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `chv_ekaterina`                              | `chv` (Chuvash)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `erz_alexandr`                               | `erz` (Erzya)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_base`, `v5_cis_base_nostress` | `sah_zinaida`                                | `sah` (Yakut)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |

#### V5 CIS Ext Models

- All of the below models support `8000`, `24000`, `48000` sampling rates and contain no auto-stress or homographs;
- `v5_cis_ext` models assume that proper stress should be added for each word for all languages, i.e. `к+ошка`;
- `v5_cis_ext_nostress` are coming soon;
- All of the below models are published under `CC-NC-BY` licence;
- V5 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

| ID           | Speakers                                                              | Language          | Colab |
| ------------ | --------------------------------------------------------------------- | ----------------- | -------------------- |
| `v5_cis_ext` | `kaz_abai`, `kaz_aidana`, `kaz_aisha`, `kaz_bakir`, `kaz_danara`      | `kaz` (Kazakh)    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `xal_delghir`, `xal_erdni`                                            | `xal` (Kalmyk)    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `tat_adiba`, `tat_alsou`, `tat_amir`, `tat_azat`, `tat_batir`         | `tat` (Tatar)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `tat_bulat`, `tat_damir`, `tat_guzel`, `tat_ildar`, `tat_ilgiz`       | `tat` (Tatar)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `tat_karim`, `tat_mansur`, `tat_murat`, `tat_rasima`, `tat_rustem`    | `tat` (Tatar)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `tat_timur`, `tat_zifa`, `tat_zufar`, `tat_zulfiya`                   | `tat` (Tatar)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `uzb_anora`, `uzb_dilnavoz`                                           | `uzb` (Uzbek)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `ukr_kateryna`, `ukr_lada`, `ukr_mykyta`, `ukr_oleksa`, `ukr_tetiana` | `ukr` (Ukrainian) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |
| `v5_cis_ext` | `chv_aihwa`, `chv_alima`                                              | `chv` (Chuvash)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts_cis.ipynb) |

#### V4

V4 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

<details>
  <summary>V4 models: v4_ru, v4_cyrillic, v4_ua, v4_uz, v4_indic </summary>


| ID       | Speakers |Auto-stress | Language                           | SR              | Colab                                                                                                                                                                        |
| ------------- | ----------- | ----------- |---------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v4_ru`    | `aidar`, `baya`, `kseniya`, `xenia`, `eugene`, `random` | yes  | `ru` (Russian)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| [`v4_cyrillic`](#cyrillic-languages)   | `b_ava`, `marat_tt`, `kalmyk_erdni`...             | no   | `cyrillic` [(Avar, Tatar, Kalmyk, ...)](#cyrillic-languages)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v4_ua`    | `mykyta`, `random`                                        | no   | `ua` (Ukrainian) | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v4_uz`    | `dilnavoz`                                                | no   | `uz` (Uzbek)     | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| [`v4_indic`](#indic-languages)   | `hindi_male`, `hindi_female`, ..., `random`             | no   | `indic` [(Hindi, Telugu, ...)](#indic-languages)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
 
</details>

#### V3

V3 models support [SSML](https://github.com/snakers4/silero-models/wiki/SSML). Also see Colab examples for main SSML tag usage.

<details>
  <summary>V3 models: v3_en, v3_en_indic, v3_de, v3_es, v3_fr, v3_indic </summary>


| ID       | Speakers |Auto-stress | Language                           | SR              | Colab                                                                                                                                                                        |
| ------------- | ----------- | ----------- |---------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v3_en`    | `en_0`, `en_1`, ..., `en_117`, `random`                   | no   | `en` (English)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_en_indic`   | `tamil_female`, ..., `assamese_male`, `random`       | no   | `en` (English)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_de`    | `eva_k`, ..., `karlsson`, `random`                        | no   | `de` (German)    | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_es`    | `es_0`, `es_1`, `es_2`, `random`                          | no   | `es` (Spanish)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| `v3_fr`    | `fr_0`, ..., `fr_5`, `random`                             | no   | `fr` (French)    | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |
| [`v3_indic`](#indic-languages)   | `hindi_male`, `hindi_female`, ..., `random`             | no   | `indic` [(Hindi, Telugu, ...)](#indic-languages)   | `8000`, `24000`, `48000` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb) |

</details>


### Dependencies

Basic dependencies for Colab examples:

- `torch`, 1.10+ for v3 models/ 2.0+ for v4 and v5 models;
- `torchaudio`, latest version bound to PyTorch should work (required only because models are hosted together with STT, not required for work);
- `omegaconf`,  latest (can be removed as well, if you do not load all of the configs);

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb)

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-models_tts/)

```python
# V5
import torch

language = 'ru'
model_id = 'v5_ru'
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

- Standalone usage only requires PyTorch 1.12+ and the Python Standard Library;
- Please see the detailed examples in Colab;

```python
# V5
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v5_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности!'
sample_rate = 48000
speaker='baya'

audio_paths = model.save_wav(text=example_text,
                             speaker=speaker,
                             sample_rate=sample_rate)
```

### SSML

Check out our [TTS Wiki page.](https://github.com/snakers4/silero-models/wiki/SSML)

### Cyrillic languages v4

> To be superseded with v5 model(s) soon.

Supported tokenset:
`!,-.:?iµöабвгдежзийклмнопрстуфхцчшщъыьэюяёђѓєіјњћќўѳғҕҗҙқҡңҥҫүұҳҷһӏӑӓӕӗәӝӟӥӧөӱӳӵӹ `

| Speaker_ID   | Language        | Gender |
| ------------ | --------------- | ------ |
| b_ava        | Avar            | F      |
| b_bashkir    | Bashkir         | M      |
| b_bulb       | Bulgarian       | M      |
| b_bulc       | Bulgarian       | M      |
| b_che        | Chechen         | M      |
| b_cv         | Chuvash         | M      |
| cv_ekaterina | Chuvash         | F      |
| b_myv        | Erzya           | M      |
| b_kalmyk     | Kalmyk          | M      |
| b_krc        | Karachay-Balkar | M      |
| kz_M1        | Kazakh          | M      |
| kz_M2        | Kazakh          | M      |
| kz_F3        | Kazakh          | F      |
| kz_F1        | Kazakh          | F      |
| kz_F2        | Kazakh          | F      |
| b_kjh        | Khakas          | F      |
| b_kpv        | Komi-Ziryan     | M      |
| b_lez        | Lezghian        | M      |
| b_mhr        | Mari            | F      |
| b_mrj        | Mari High       | M      |
| b_nog        | Nogai           | F      |
| b_oss        | Ossetic         | M      |
| b_ru         | Russian         | M      |
| b_tat        | Tatar           | M      |
| marat_tt     | Tatar           | M      |
| b_tyv        | Tuvinian        | M      |
| b_udm        | Udmurt          | M      |
| b_uzb        | Uzbek           | M      |
| b_sah        | Yakut           | M      |
| kalmyk_erdni | Kalmyk          | M      |
| kalmyk_delghir | Kalmyk        | F      |

### Indic languages v4

#### Example

(!!!) All input sentences should be romanized to ISO format using [`aksharamukha`](https://aksharamukha.appspot.com/python). An example for `hindi`:

```python
# V3
import torch
from aksharamukha import transliterate

# Loading model
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v4_indic')

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

## Contact

Try our models, create an [issue](https://github.com/snakers4/silero-models/issues/new), join our [chat](https://t.me/silero_speech), [email](mailto:hello@silero.ai) us, and read the latest [news](https://t.me/silero_news).

## Licence

All of the models are published under the main repo license (i.e. CC-NC-BY) except for the `base` cis-tts models, which are under MIT.

## Citations

```bibtex
@misc{Silero Models,
  author = {Silero Team},
  title = {Silero Models: pre-trained text-to-speech models made embarrassingly simple},
  year = {2025},
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
  - Multilingual Text-to-Speech Models for Indic Languages - [link](https://www.analyticsvidhya.com/blog/2022/06/multilingual-text-to-speech-models-for-indic-languages/)
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
  - OpenAI решили распознавание речи! Разбираемся так ли это … - [link](https://habr.com/ru/post/689572/)
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
  - Speech-To-Text - [link](https://www.silero.ai/tag/speech-to-text/)

- TTS:
  - Делаем быстрый, качественный и доступный синтез на языках России — нужно ваше участие - [link](https://habr.com/ru/articles/872474/)
  - Мы решили задачу омографов и ударений в русском языке - [link](https://habr.com/ru/articles/955130/)
  - Теперь наш синтез также доступен в виде бота в Телеграме - [link](https://habr.com/ru/post/682188/)
  - Может ли синтез речи обмануть систему биометрической идентификации? - [link](https://habr.com/ru/post/673996/)
  - Теперь наш синтез на 20 языках - [link](https://habr.com/ru/post/669910/)
  - Теперь наш публичный синтез в супер-высоком качестве, в 10 раз быстрее и без детских болячек - [link](https://habr.com/ru/post/660565/)
  - Синтезируем голос бабушки, дедушки и Ленина + новости нашего публичного синтеза - [link](https://habr.com/ru/post/584750/)
  - Мы сделали наш публичный синтез речи еще лучше - [link](https://habr.com/ru/post/563484/)
  - Мы Опубликовали Качественный, Простой, Доступный и Быстрый Синтез Речи - [link](https://habr.com/ru/post/549480/)

- VAD:
  - Новый релиз публичного детектора голоса Silero VAD v6 - [link](https://habr.com/ru/articles/940750/)
  - Наш публичный детектор голоса стал лучше - [link](https://habr.com/ru/post/695738/)
  - А ты используешь VAD? Что это такое и зачем он нужен - [link](https://habr.com/ru/post/594745/)
  - Модели для Детекции Речи, Чисел и Распознавания Языков - [link](https://www.silero.ai/vad-lang-classifier-number-detector/)
  - Мы опубликовали современный Voice Activity Detector и не только -[link](https://habr.com/ru/post/537274/)

- Text Enhancement:
  - Восстановление знаков пунктуации и заглавных букв — теперь и на длинных текстах - [link](https://habr.com/ru/post/594565/)
  - Мы опубликовали модель, расставляющую знаки препинания и заглавные буквы в тексте на четырех языках - [link](https://habr.com/ru/post/581946/)
