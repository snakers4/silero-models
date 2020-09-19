#
# a modified version of this script https://github.com/magenta/ddsp/blob/master/ddsp/colab/colab_utils.py
# modified in line with the rest of examples code
#

import io
import base64
import tempfile
from typing import Optional
from utils import read_audio
from pydub import AudioSegment
from google.colab import files
from google.colab import output
from IPython import display as _display


def record_audio(seconds: int = 3,
                 normalize_db: float = 0.1):
    # Use Javascript to record audio.
    record_js_code = """
      const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
      const b2text = blob => new Promise(resolve => {
        const reader = new FileReader()
        reader.onloadend = e => resolve(e.srcElement.result)
        reader.readAsDataURL(blob)
      })
      var record = time => new Promise(async resolve => {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        recorder = new MediaRecorder(stream)
        chunks = []
        recorder.ondataavailable = e => chunks.push(e.data)
        recorder.start()
        await sleep(time)
        recorder.onstop = async ()=>{
          blob = new Blob(chunks)
          text = await b2text(blob)
          resolve(text)
        }
        recorder.stop()
      })
      """
    print('Starting recording for {} seconds...'.format(seconds))
    _display.display(_display.Javascript(record_js_code))
    audio_string = output.eval_js('record(%d)' % (seconds * 1000.0))
    print('Finished recording!')
    audio_bytes = base64.b64decode(audio_string.split(',')[1])
    return audio_bytes_to_np(audio_bytes,
                             normalize_db=normalize_db)


def audio_bytes_to_np(wav_data: bytes,
                      normalize_db: float = 0.1):
    # Parse and normalize the audio.
    audio = AudioSegment.from_file(io.BytesIO(wav_data))
    audio.remove_dc_offset()
    if normalize_db is not None:
        audio.normalize(headroom=normalize_db)
    # Save to tempfile and load with librosa.
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav_file:
        fname = temp_wav_file.name
        audio.export(fname, format='wav')
        wav = read_audio(fname)
    return wav


def upload_audio(normalize_db: Optional[float] = None):
    audio_files = files.upload()
    fnames = list(audio_files.keys())
    if len(fnames) == 0:
        return None
    return read_audio(fnames[0])
