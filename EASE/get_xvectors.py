from speechbrain.pretrained import EncoderClassifier
import os
import torchaudio
import numpy as np
from tqdm import tqdm
import pandas as pd

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})


folder = "/path/to/wav/files/"
wav_files = os.listdir(folder)
wav_files = [x for x in wav_files if ".wav" in x]
target_folder = "/target/path/for/x-vectors/"
os.makedirs(target_folder, exist_ok=True)


for i, wav_file in enumerate(tqdm(wav_files)):
    target_file = os.path.join(target_folder, wav_file.split(os.sep)[-1].replace(".wav", ".npy"))
    if os.path.exists(target_file) == True:
        continue
    wav_file = os.path.join(folder, wav_file)
    sig, sr = torchaudio.load(wav_file)
    embeddings = classifier.encode_batch(sig.cuda())[0, 0, :]
    
    np.save(target_file, embeddings.cpu().detach().numpy())
