dependencies = ['torch', 'omegaconf', 'torchaudio']
from omegaconf import OmegaConf
from utils import init_jit_model


def silero_stt(language='en', **kwargs):
    """ Silero Speech-To-Text Model(s)
    language (str): language of the model, now available are ['en', 'de', 'es']
    Returns a model and a decoder object
    Please see https://github.com/snakers4/silero-models for usage examples
    """
    models = OmegaConf.load('models.yml')
    available_languages = list(models.stt_models.keys())
    assert language in available_languages
    return init_jit_model(model_url=models.stt_models.get('en').latest.jit,
                          **kwargs)
