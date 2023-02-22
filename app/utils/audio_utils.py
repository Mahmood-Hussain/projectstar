import soundfile

def write_wav(path, audio, sr=8000):
    soundfile.write(path, audio, sr, "PCM_16")