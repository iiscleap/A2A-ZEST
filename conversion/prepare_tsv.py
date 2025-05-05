import pandas as pd
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Pitch")

parser.add_argument(
    "--input_folder",
    metavar="input_folder",
    type=str
)
parser.add_argument(
    "--out_file",
    metavar="out_file",
    type=str
)
args = parser.parse_args()

# test_dict = json.load(open("/data1/soumyad/multilingual_ZEST/test_labels.json", "r"))
# transcripts = json.load(open("/data1/soumyad/multilingual_ZEST/evaluation/esd_transcripts.json", "r"))
# df_metadata = pd.read_csv("/data1/soumyad/multilingual_ZEST/multilingual_data/metadata.csv")
# df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="test")]
# files = df_metadata["path"].values
# df = {"text":[], "audio_path":[]}

# for k in files:
#     path = os.path.join("/data1/soumyad/multilingual_ZEST/multilingual_data/wavs/", k)
#     text = transcripts[k.split("_")[-1].replace(".wav", "")]
#     df["text"].append(text)
#     df["audio_path"].append(path)

# df = pd.DataFrame(df)
# df.to_csv("english_test.tsv", sep="\t", index=False)

transcripts = json.load(open("/home/soumyadutta/ZEST/TIMIT/timit_transcripts.json", "r"))
folder = args.input_folder
converted_files = os.listdir(folder)
converted_files = [x for x in converted_files if ".wav" in x]
df = {"text":[], "audio_path":[]}
for k in converted_files:
    path = os.path.join(folder, k)
    # text = transcripts[k.split("--")[0].split("_")[1]]
    text = transcripts[k.split("--")[0]+".wav"]
    df["text"].append(text)
    df["audio_path"].append(path)
df = pd.DataFrame(df)
df.to_csv(args.out_file, sep="\t", index=False)