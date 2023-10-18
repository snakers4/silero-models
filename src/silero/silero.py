import os
import torch


def silero_stt(language='en',
               version='latest',
               jit_model='jit',
               **kwargs):
    """ Silero Speech-To-Text Model(s)
    language (str): language of the model, now available are ['en', 'de', 'es']
    Returns a model, decoder object and a set of utils
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    from omegaconf import OmegaConf    
    from .utils import (init_jit_model,
                        read_audio,
                        read_batch,
                        split_into_batches,
                        prepare_model_input)

    models_list_file = os.path.join(os.path.dirname(__file__), "..", "..", "models.yml")
    if not os.path.exists(models_list_file):
        models_list_file = 'latest_silero_models.yml'
    if not os.path.exists(models_list_file):
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                    'latest_silero_models.yml',
                                    progress=False)
    assert os.path.exists(models_list_file)
    models = OmegaConf.load(models_list_file)
    available_languages = list(models.stt_models.keys())
    assert language in available_languages

    model, decoder = init_jit_model(model_url=models.stt_models.get(language).get(version).get(jit_model),
                                    **kwargs)
    utils = (read_batch,
             split_into_batches,
             read_audio,
             prepare_model_input)

    return model, decoder, utils


def silero_tts(language='en',
               speaker='kseniya_16khz',
               **kwargs):
    """ Silero Text-To-Speech Models
    language (str): language of the model, now available are ['ru', 'en', 'de', 'es', 'fr']
    Returns a model and a set of utils
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    from omegaconf import OmegaConf      
    from .tts_utils import apply_tts
    from .tts_utils import init_jit_model as init_jit_model_tts

    models_list_file = os.path.join(os.path.dirname(__file__), "..", "..", "models.yml")
    if not os.path.exists(models_list_file):
        models_list_file = 'latest_silero_models.yml'
    if not os.path.exists(models_list_file):
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                    'latest_silero_models.yml',
                                    progress=False)
    assert os.path.exists(models_list_file)
    models = OmegaConf.load(models_list_file)
    available_languages = list(models.tts_models.keys())
    assert language in available_languages, f'Language not in the supported list {available_languages}'
    available_speakers = []
    speaker_language = {}
    for lang in available_languages:
        speakers = list(models.tts_models.get(lang).keys())
        available_speakers.extend(speakers)
        for _ in speakers:
            speaker_language[_] = lang
    assert speaker in available_speakers, f'Speaker not in the supported list {available_speakers}'
    assert language == speaker_language[speaker], f"Incorrect language '{language}' for this speaker, please specify '{speaker_language[speaker]}'"

    model_conf = models.tts_models[language][speaker].latest
    if '_v2' in speaker or '_v3' in speaker or 'v3_' in speaker or 'v4_' in speaker:
        from torch import package
        model_url = model_conf.package
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, os.path.basename(model_url))
        if not os.path.isfile(model_path):
            torch.hub.download_url_to_file(model_url,
                                           model_path,
                                           progress=True)
        imp = package.PackageImporter(model_path)
        model = imp.load_pickle("tts_models", "model")
        if speaker == 'multi_v2':
            avail_speakers = model_conf.speakers
            return model, avail_speakers
        else:
            example_text = model_conf.example
            return model, example_text
    else:
        model = init_jit_model_tts(model_conf.jit)
        symbols = model_conf.tokenset
        example_text = model_conf.example
        sample_rate = model_conf.sample_rate
        return model, symbols, sample_rate, example_text, apply_tts


def silero_te():
    """ Silero Texts Enhancing Models
    Current model supports the following languages: ['en', 'de', 'ru', 'es']
    Returns a model and a set of utils
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    import yaml
    from torch import package

    models_list_file = os.path.join(os.path.dirname(__file__), "..", "..", "models.yml")
    if not os.path.exists(models_list_file):
        models_list_file = 'latest_silero_models.yml'
    if not os.path.exists(models_list_file):
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                    'latest_silero_models.yml',
                                    progress=False)

    assert os.path.exists(models_list_file)
    with open(models_list_file, 'r', encoding='utf8') as yaml_file:
        models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    model_conf = models.get('te_models').get('latest')

    model_url = model_conf.get('package')
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))

    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url,
                                       model_path,
                                       progress=True)

    imp = package.PackageImporter(model_path)
    model = imp.load_pickle("te_model", "model")
    example_texts = model.examples
    languages = model_conf.get('languages')
    punct = model_conf.get('punct')

    def apply_te(text, lan='en'):
        return model.enhance_text(text, lan)

    return (model,
            example_texts,
            languages,
            punct,
            apply_te)


def silero_denoise(
    name='small_slow',
    version='latest',
    **kwargs
):
    """ Silero Denoise Model(s)
    name (str): model name, available names (in order of best outputs to worst): ['small_slow', 'large_fast', 'small_fast']
    version (str): model version
    Returns a model and a set of utils
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    from omegaconf import OmegaConf    
    from .denoiser_utils import (
        init_jit_model,
        read_audio,
        save_audio,
        denoise
    )

    models_list_file = os.path.join(os.path.dirname(__file__), "..", "..", "models.yml")
    if not os.path.exists(models_list_file):
        models_list_file = 'latest_silero_models.yml'
    if not os.path.exists(models_list_file):
        torch.hub.download_url_to_file(
            'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
            'latest_silero_models.yml',
            progress=False
        )
    assert os.path.exists(models_list_file)
    models = OmegaConf.load(models_list_file)
    available_models = models.denoise_models.models
    samples = models.denoise_models.samples
    assert name in available_models

    model = init_jit_model(model_url=models.denoise_models.get(name).get(version).get('jit'),
                                    **kwargs)
    utils = (
        read_audio,
        save_audio,
        denoise
    )

    return model, samples, utils
