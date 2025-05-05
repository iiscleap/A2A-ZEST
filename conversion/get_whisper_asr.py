import whisper
import os
from tqdm import tqdm
import torch
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Pitch")
parser.add_argument(
    "--input_folder",
    metavar="input_folder",
    type=str,
)
parser.add_argument(
    "--out_file",
    metavar="out_file",
    type=str,
)

args = parser.parse_args()
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


wav_folder = args.input_folder
wavs = os.listdir(wav_folder)
wavs = [x for x in wavs if ".wav" in x]

# wav_folder = "/data1/soumyad/multilingual_ZEST/wavs"
# df_metadata = pd.read_csv("/home/soumyadutta/ZEST/evaluation/metadata.csv")
# df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="test")]
# wavs = df_metadata["path"].values
# wavs = [x.split(os.sep)[-1] for x in wavs] 

model = whisper.load_model("large-v3", download_root = "/home/soumyadutta/soumyad/whisper-models/")
model = model.to(device)

transcripts = {}
for i, wav in enumerate(tqdm(wavs)):
    wav_name = os.path.join(wav_folder, wav)
    audio = whisper.load_audio(wav_name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    transcripts[wav] = result.text

with open(args.out_file, "w") as f:
    json.dump(transcripts, f)