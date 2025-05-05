from speechbrain.pretrained import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Pitch")

parser.add_argument(
    "--converted_folder",
    metavar="converted_folder",
    type=str
)
parser.add_argument(
    "--target_folder",
    metavar="target_folder",
    type=str
)
args = parser.parse_args()

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
# df_metadata = pd.read_csv("/home/soumyadutta/ZEST/wavs/metadata.csv")
# df_metadata = df_metadata[(df_metadata["language"]=="english")]
# wav_files = df_metadata["path"].values
folder = args.converted_folder
wav_files = os.listdir(folder)
wav_files = [x for x in wav_files if ".wav" in x]
# files = [x.split(os.sep)[-1] for x in files]

# folder = "/data1/soumyad/multilingual_ZEST/wavs"
target_folder = args.target_folder
os.makedirs(target_folder, exist_ok=True)
# wav_files = os.listdir(folder)
# wav_files = [x for x in wav_files if ".wav" in x]
# wav_files = [x for x in wav_files if ".npy" not in x]

for i, wav_file in enumerate(tqdm(wav_files)):
    target_file = os.path.join(target_folder, wav_file.split(os.sep)[-1].replace(".wav", ".npy"))
    if os.path.exists(target_file) == True:
        continue
    sig, sr = torchaudio.load(os.path.join(folder, wav_file))
    embeddings = classifier.encode_batch(sig.cuda())[0, 0, :]
    
    np.save(target_file, embeddings.cpu().detach().numpy())