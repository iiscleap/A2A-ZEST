import whisper
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
import os
from tqdm import tqdm
import torch
import pickle5 as pickle
import argparse
import pandas as pd
import re
import unicodedata
import regex
from evaluate import load
import json

parser = argparse.ArgumentParser(description="get wer")
parser.add_argument(
    "--gt_file",
    type=str,
)
parser.add_argument(
    "--gen_file",
    type=str,
)
args = parser.parse_args()
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text_normalizer = EnglishTextNormalizer()
wer = load("wer")
cer = load("cer")

gt_transcripts = json.load(open("/home/soumyadutta/ZEST/TIMIT/timit_transcripts.json", "r"))
gen_transcripts = json.load(open(args.gen_file, "r"))

# df_metadata = pd.read_csv("/home/soumyadutta/ZEST/evaluation/metadata.csv")
# df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="test")]
# files = df_metadata["path"].values
# files = [x.split(os.sep)[-1] for x in files]
# labels = df_metadata["emotion"].values 
# label_dict = dict(zip(files, labels))

gt, gen = [], []
for i, (k,v) in enumerate(tqdm(gen_transcripts.items())):
    # if label_dict[k] == "neutral":
    # if k.split("_")[-1] != "001428.wav":
    #     continue
    gt.append(text_normalizer(gt_transcripts[k.split("--")[0]+".wav"]))
    # gt.append(text_normalizer(gt_transcripts[k.split("--")[0].split("_")[-1]]))
    # gt.append(text_normalizer(gt_transcripts[k.replace(".WAV", ".wav")]))
    gen.append(text_normalizer(gen_transcripts[k]))
print(len(gt))
results = wer.compute(predictions=gen, references=gt)
print("WER", results)
results = cer.compute(predictions=gen, references=gt)
print("CER", results)

