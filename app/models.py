import torch
import librosa
import os

from methods.UNet1D import UNet1D, write_wav
import __main__


def run_model_on_file(file_path):
    # create instance of UNet1D model
    setattr(__main__, "UNet1D", UNet1D)
    model = UNet1D() # unnecessary, but here for removing error
    # load the model from static/model_files/unet1d_model_epoch20.pt
    model = torch.load(os.getenv('UNET1D_CHECKPOINT'), map_location=torch.device('cpu'))
    # load the audio file
    audio, sr = librosa.load(file_path, sr=8000, mono=True, res_type='kaiser_fast')
    # convert the audio to a tensor
    audio_tensor = torch.from_numpy(audio).float()
    # add a batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)
    # add a channel dimension
    audio_tensor = audio_tensor.unsqueeze(0)
    # pass the audio to the model
    output = model(audio_tensor)
    # convert the output to a numpy array
    output = output.detach().numpy()
    # remove the channel dimension
    output = output.squeeze(0)
    # remove the batch dimension
    output = output.squeeze(0)
    # save the output to a wav file
    os.makedirs(os.getenv('CONVERTED_DIR'), exist_ok=True)
    # generate a random name for the output file
    name = os.urandom(24).hex()
    write_wav(os.getenv('CONVERTED_DIR') + '/' + name + '.wav', output, sr)
    # return the complete path to the output file
    return name + '.wav'

