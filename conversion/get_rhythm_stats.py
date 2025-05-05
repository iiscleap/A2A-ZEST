import pandas as pd
import json
import os
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description="Pitch")

parser.add_argument(
    "--input_file",
    metavar="input_file",
    type=str
)
args = parser.parse_args()
test_df = pd.read_csv("/home/soumyadutta/ZEST/evaluation/english_test_rate.tsv", sep="\t")

# test_df = pd.read_csv("/home/soumyadutta/ZEST/CREMA/crema_rate.tsv", sep="\t")
test_rates = {"aud":[], "word":[], "syllable":[], "char":[], "phoneme":[], "vowel":[]}
utterances = list(test_df["utterance"])
speech_rate_word = list(test_df["speech_rate_word"])
speech_rate_syllable = list(test_df["speech_rate_syllable"])
speech_rate_char = list(test_df["speech_rate_char"])
speech_rate_phoneme = list(test_df["speech_rate_phoneme"])
speech_rate_vowel = list(test_df["speech_rate_vowel"])
utterances = [json.loads(x) for x in utterances]
utterances = [x["id"] for x in utterances]
for i in range(len(utterances)):
    test_rates["aud"].append(utterances[i].split(os.sep)[-1])
    test_rates["word"].append(speech_rate_word[i])
    test_rates["syllable"].append(speech_rate_syllable[i])
    test_rates["char"].append(speech_rate_char[i])
    test_rates["phoneme"].append(speech_rate_phoneme[i])
    test_rates["vowel"].append(speech_rate_vowel[i])

converted_df = pd.read_csv(args.input_file, sep="\t")
converted_rates = {"aud":[], "word":[], "syllable":[], "char":[], "phoneme":[], "vowel":[]}
utterances = list(converted_df["utterance"])
speech_rate_word = list(converted_df["speech_rate_word"])
speech_rate_syllable = list(converted_df["speech_rate_syllable"])
speech_rate_char = list(converted_df["speech_rate_char"])
speech_rate_phoneme = list(converted_df["speech_rate_phoneme"])
speech_rate_vowel = list(converted_df["speech_rate_vowel"])
utterances = [json.loads(x) for x in utterances]
utterances = [x["id"] for x in utterances]
for i in range(len(utterances)):
    converted_rates["aud"].append(utterances[i].split(os.sep)[-1])
    converted_rates["word"].append(speech_rate_word[i])
    converted_rates["syllable"].append(speech_rate_syllable[i])
    converted_rates["char"].append(speech_rate_char[i])
    converted_rates["phoneme"].append(speech_rate_phoneme[i])
    converted_rates["vowel"].append(speech_rate_vowel[i])

target_word, target_syllable, target_char, target_phoneme, target_vowel = [], [], [], [], []
pred_word, pred_syllable, pred_char, pred_phoneme, pred_vowel = [], [], [], [], []

for i, f in enumerate(converted_rates["aud"]):
    j = test_rates["aud"].index(f.split("--")[-1])
    target_word.append(test_rates["word"][j])
    target_syllable.append(test_rates["syllable"][j])
    target_char.append(test_rates["char"][j])
    target_phoneme.append(test_rates["phoneme"][j])
    target_vowel.append(test_rates["vowel"][j])

    pred_word.append(converted_rates["word"][i])
    pred_syllable.append(converted_rates["syllable"][i])
    pred_char.append(converted_rates["char"][i])
    pred_phoneme.append(converted_rates["phoneme"][i])
    pred_vowel.append(converted_rates["vowel"][i])

res = stats.pearsonr(pred_word, target_word)
print("Word", res.statistic)
res = stats.pearsonr(pred_syllable, target_syllable)
print("Syllable", res.statistic)
res = stats.pearsonr(pred_char, target_char)
print("Char", res.statistic)
res = stats.pearsonr(pred_phoneme, target_phoneme)
print("Phoneme", res.statistic)
res = stats.pearsonr(pred_vowel, target_vowel)
print("Vowel", res.statistic)