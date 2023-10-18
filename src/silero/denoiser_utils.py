import os
import torch
import torchaudio


def read_audio(
    path: str,
    sampling_rate: int = 24000
):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=sampling_rate
        )
        wav = transform(wav)
        sr = sampling_rate
    assert sr == sampling_rate
    return wav * 0.95


def save_audio(
    path: str,
    tensor: torch.Tensor,
    sampling_rate: int = 48000
):
    torchaudio.save(path, tensor, sampling_rate)


def init_jit_model(
    model_url: str,
    device: torch.device = torch.device('cpu')
):
    torch._C._jit_set_profiling_mode(False) 
    torch.set_grad_enabled(False)
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))
    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url,
                                       model_path,
                                       progress=True)
    model = torch.jit.load(model_path)
    model.eval()
    model = model.to(device)
    return model


def denoise(
    model: torch.nn.Module,
    audio_path: str,
    save_path: str = 'result.wav',
    device=torch.device('cpu'),
    sampling_rate: int = 48000
):
    audio = read_audio(audio_path).to(device)
    model.to(device)
    out = model(audio).flatten().unsqueeze(0)
    if save_path:
        save_audio(save_path, out.cpu(), sampling_rate)
    return out, sampling_rate
