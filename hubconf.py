dependencies = ['torch', 'omegaconf', 'torchaudio']
import torch
from omegaconf import OmegaConf
from utils import (init_jit_model,
                   read_audio,
                   read_batch,
                   split_into_batches,
                   prepare_model_input)


def silero_stt(language='en', **kwargs):
    """ Silero Speech-To-Text Model(s)
    language (str): language of the model, now available are ['en', 'de', 'es']
    Returns a model, decoder object and a set of utils
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                   'latest_silero_models.yml',
                                   progress=False)
    models = OmegaConf.load('latest_silero_models.yml')
    available_languages = list(models.stt_models.keys())
    assert language in available_languages

    model, decoder = init_jit_model(model_url=models.stt_models.get(language).latest.jit,
                                    **kwargs)
    utils = (read_batch,
             split_into_batches,
             read_audio,
             prepare_model_input)

    return model, decoder, utils
