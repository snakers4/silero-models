import torch
import tempfile
import warnings
import torchaudio
from typing import List
from itertools import groupby


torchaudio.set_audio_backend("soundfile")  # switch backend


def read_batch(audio_paths: List[str]):
    return [read_audio(audio_path)
            for audio_path
            in audio_paths]


def split_into_batches(lst: List[str],
                       batch_size: int = 10):
    return [lst[i:i + batch_size]
            for i in
            range(0, len(lst), batch_size)]


def read_audio(path: str,
               target_sr: int = 16000):

    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path,
                              normalization=True,
                              channels_first=True)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)


def prepare_model_input(batch: List[torch.Tensor],
                        device=torch.device('cpu')):
    max_seqlength = max(max([len(_) for _ in batch]), 12800)
    inputs = torch.zeros(len(batch), max_seqlength)
    for i, wav in enumerate(batch):
        inputs[i, :len(wav)].copy_(wav)
    inputs = inputs.to(device)
    return inputs


class Decoder():
    def __init__(self,
                 labels: List[str]):
        self.labels = labels
        self.blank_idx = self.labels.index('_')

    def process(self,
                probs):
        assert len(self.labels) == probs.shape[1]
        for_string = []
        argm = torch.argmax(probs, axis=1)
        for i in argm:
            if i == self.labels.index('2'):
                try:
                    prev = for_string[-1]
                    for_string.append('$')
                    for_string.append(prev)
                    continue
                except:
                    for_string.append(' ')
                    warnings.warn('Token "2" detected a the beginning of sentence, omitting')
            if i != self.blank_idx:
                for_string.append(self.labels[i])
        string = ''.join([x[0] for x in groupby(for_string)]).replace('$', '').strip()
        return string

    def __call__(self,
                 probs: torch.Tensor):
        return self.process(probs)


def init_jit_model(model_url: str,
                   device: torch.device = torch.device('cpu')):
    torch.set_grad_enabled(False)
    with tempfile.NamedTemporaryFile('wb', suffix='.model') as f:
        torch.hub.download_url_to_file(model_url,
                                       f.name,
                                       progress=True)
        model = torch.jit.load(f.name, map_location=device)
        model.eval()
    return model, Decoder(model.labels)
