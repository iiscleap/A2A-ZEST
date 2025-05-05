# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path
import torch
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write
import soundfile as sf
from dataset import CodeDataset, parse_manifest, mel_spectrogram, \
    MAX_WAV_VALUE, load_audio
from utils import AttrDict
from bigvgan import CodeGenerator
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from librosa.util import normalize
import pickle
from tqdm import tqdm
import ast

h = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * 32768
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf

def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global f0_stats
    global spkrs_emb
    global dataset
    global spkr_dataset
    global idx
    global device
    global a
    global h
    global spkrs

    a = arguments
    idx = queue.get()
    device = idx

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])


    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                              f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                              f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                              f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                              pad=a.pad, pitch_folder=a.pitch_folder, emo_folder=a.emo_folder)

    if a.unseen_f0:
        dataset.f0_stats = torch.load(a.unseen_f0)

    os.makedirs(a.output_dir, exist_ok=True)

    if h.get('multispkr', None):
        spkrs = random.sample(range(len(dataset.id_to_spkr)), k=min(5, len(dataset.id_to_spkr)))

    if a.f0_stats and h.get('f0', None) is not None:
        f0_stats = torch.load(a.f0_stats)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def inference(item_index):

    code, gt_audio, filename, _ = dataset[item_index]
    source_file = filename.split(os.sep)[-1].replace(".flac", "")
    code = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in code.items()}

    if a.parts:
        parts = Path(filename).parts
        fname_out_name = '_'.join(parts[-3:])[:-4]
    else:
        fname_out_name = Path(filename).stem
   
    f0_stats = torch.load("/home/soumyad/unit2wav/speech-resynthesis/datasets/ESD/esd_f0_stats.pth")

    #if int(fname_out_name[5:11]) < 350:
    if True:  
        if h.get('f0_vq_params', None) or h.get('f0_quantizer', None):
            to_remove = gt_audio.shape[-1] % (16 * 80)
            assert to_remove % h['code_hop_size'] == 0

            if to_remove != 0:
                to_remove_code = to_remove // h['code_hop_size']
                to_remove_f0 = to_remove // 320

                gt_audio = gt_audio[:-to_remove]
                code['code'] = code['code'][..., :-to_remove_code]
                code['f0'] = code['f0'][..., :-to_remove_f0]

        new_code = dict(code)
        if 'f0' in new_code:
            del new_code['f0']
            new_code['f0'] = code['f0']
        # src_f0 = new_code['f0'].clone().cpu().numpy()
        # audio, rtf = generate(h, generator, new_code)
        # output_file = os.path.join(a.output_dir, fname_out_name + '.wav')
        # audio = librosa.util.normalize(audio.astype(np.float32))
        # write(output_file, h.sampling_rate, audio)
        
        if h.get('multispkr', None) and a.convert:
            reference_folder = "/home/soumyad/TIMIT/esd_files"
            f0_folder = "/home/soumyad/Librispeech/predf0_iemocap/"
            tot_files = os.listdir(f0_folder)
            for i, filename in enumerate(tot_files):
                if source_file not in filename:
                    continue
                emo_file = filename.split("-")[-1][4:]
                emo_embed = np.load("/home/soumyad/DISSC/wav2vec_feats_iemocap/" + emo_file)
                feats = {}
                f0 = np.load(f0_folder + filename)
                f0 = f0.astype(np.float32)
                trg_f0 = f0
                new_f0 = torch.tensor(f0)
                new_f0 = new_f0.squeeze(-1)
                code['f0'] = torch.FloatTensor(new_f0).to(device)
                code['f0'] = code['f0'].unsqueeze(0).unsqueeze(0)
                if code['f0'].shape[-1] > new_code['f0'].shape[-1]:
                    code['f0'] = code['f0'][:, :, :new_code['f0'].shape[-1]]

                # audio, sampling_rate = load_audio(os.path.join("/home/soumyad/TIMIT/esd_files", filename))
                # audio = audio / MAX_WAV_VALUE
                # audio = normalize(audio) * 0.95
                # audio = torch.FloatTensor(audio)
                # audio = audio.unsqueeze(0)
                # try:
                #     f0 = get_yaapt_f0(audio.numpy(), rate=sampling_rate, interp=False)
                # except:
                #     f0 = np.zeros((1, 1, audio.shape[-1] // 320))
                # f0 = f0.astype(np.float32)
                # trg_f0 = f0
                # new_f0 = torch.tensor(f0)
                # new_f0 = new_f0.squeeze(-1)

                # code['f0'] = torch.FloatTensor(new_f0).to(device)
                # if code['f0'].shape[-1] > new_code['f0'].shape[-1]:
                #     code['f0'] = code['f0'][:, :, :new_code['f0'].shape[-1]]
                # else:
                #     code['f0'] = torch.cat((code['f0'], torch.zeros((1, 1, new_code['f0'].shape[-1]-code['f0'].shape[-1])).to(device)), -1)
                code["emo_embed"] = torch.tensor(emo_embed).unsqueeze(0).to(device)
                audio, rtf = generate(h, generator, code)

                output_file = os.path.join(a.output_dir, filename).replace(".npy", ".wav")
                audio = librosa.util.normalize(audio.astype(np.float32))
                write(output_file, h.sampling_rate, audio)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default=None)
    parser.add_argument('--input_code_file', default="")
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--pitch_folder', default='')
    parser.add_argument('--energy_folder', default='')
    parser.add_argument('--spkr_folder', default='')
    parser.add_argument('--emo_folder', default='')
    parser.add_argument('--lang_folder', default='')
    parser.add_argument('--target_files', default='')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--f0-stats', type=Path)
    parser.add_argument('--vc', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--random-speakers', action='store_true')
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('--unseen-f0', type=Path)
    parser.add_argument('-n', type=int, default=30000)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(1))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)
    os.makedirs(a.output_dir, exist_ok=True)
    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    generator = CodeGenerator(h).to(device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        src_files, codes = file_list
        if ".pkl" in a.target_files:
            target_files = pickle.load(open(a.target_files, "rb"))
            with torch.no_grad():
                for i, src_file in enumerate(tqdm(src_files)):
                    src_file_name = src_file.stem
                    spkr_file = os.path.join(a.spkr_folder, src_file_name+".npy")
                    spkr_embed = torch.tensor(np.load(spkr_file)).unsqueeze(0).to(device)
                    tokens = torch.tensor(codes[i]).unsqueeze(0).to(device)
                    feats = {"code":tokens, "spkr":spkr_embed}
                    for j, target_file in enumerate(target_files):
                        target_file_name = target_file.split(os.sep)[-1]
                        converted_file_name = src_file_name + "--" + target_file_name
                        pitch_file = os.path.join(a.pitch_folder, converted_file_name.replace(".wav", ".npy"))
                        emo_file = os.path.join(a.emo_folder, converted_file_name.replace(".wav", ".npy"))
                        
                        pitch = torch.tensor(np.load(pitch_file)).unsqueeze(0).unsqueeze(0).to(device)
                        if a.emo_folder != "":
                            emo_embed = torch.tensor(np.load(emo_file)).unsqueeze(0).to(device)
                            feats["emo_embed"] = emo_embed
                        feats["f0"] = pitch
                        
                        audio, rtf = generate(h, generator, feats)
                        output_file = os.path.join(a.output_dir, converted_file_name)
                        audio = librosa.util.normalize(audio.astype(np.float32))
                        write(output_file, h.sampling_rate, audio)
        else:
            target_files_orig = json.load(open(a.target_files, "r"))
            target_files = {}
            tokens_dict = {}
            with open(a.input_code_file) as f:
                lines = f.readlines()
                for l in lines:
                    d = ast.literal_eval(l)
                    name, tokens = d["audio"], d["hubert"]
                    #remove beginning of tokens till |
                    # tokens = tokens.split("|")[1]
                    tokens_l = tokens.split(" ")
                    tokens_dict[name.split(os.sep)[-1]] = np.array(tokens_l).astype(int)
            # print(len(tokens_dict))
            for k, v in target_files_orig.items():
                target_files[k.replace("/data1/soumyad/multilingual_ZEST/wavs/", "/data1/soumyad/multilingual_ZEST/multilingual_data/wavs/")] = [x.replace("/data1/soumyad/multilingual_ZEST/wavs/", "/data1/soumyad/multilingual_ZEST/multilingual_data/wavs/") for x in v]
            # print(src_files)
            with torch.no_grad():
                for i, src_file in enumerate(tqdm(target_files)):
                    
                    src_file_name = src_file.split(os.sep)[-1]
                    spkr_file = os.path.join(a.spkr_folder, src_file_name.replace(".wav", ".npy"))
                    spkr_embed = torch.tensor(np.load(spkr_file)).unsqueeze(0).to(device)
                    targets = target_files[src_file]
                    for target_file in targets:
                        target_file_name = target_file.split(os.sep)[-1]
                        converted_file_name = src_file_name.replace(".wav", "") + "--" + target_file_name
                        tokens = torch.tensor(tokens_dict[converted_file_name]).unsqueeze(0).to(device)
                        feats = {"code":tokens, "spkr":spkr_embed}
                        tgt_pitch_file = os.path.join(a.pitch_folder, target_file_name.replace(".wav", ".npy"))
                        tgt_emo_file = os.path.join(a.emo_folder, converted_file_name.replace(".wav", ".npy"))
                        pitch = torch.tensor(np.load(tgt_pitch_file)).unsqueeze(0).unsqueeze(0).to(device)
                        if pitch.shape[-1] < tokens.shape[-1]:
                            pad = torch.zeros((1, 1, tokens.shape[-1]-pitch.shape[-1])).to(device)
                            pitch = torch.cat((pitch, pad), -1)
                        elif pitch.shape[-1] > tokens.shape[-1]:
                            pitch = pitch[:, :, :tokens.shape[-1]]
                        if a.emo_folder != "":
                            emo_embed = torch.tensor(np.load(tgt_emo_file)).unsqueeze(0).to(device)
                            feats["emo_embed"] = emo_embed
                        feats["f0"] = pitch
                        # print(pitch.shape, tokens.shape)
                        if a.energy_folder != "":
                            tgt_energy_file = os.path.join(a.energy_folder, converted_file_name.replace(".wav", ".npy"))
                            energy = torch.tensor(np.load(tgt_energy_file)).unsqueeze(0).unsqueeze(0).to(device)
                            feats["energy"] = energy
                        audio, rtf = generate(h, generator, feats)
                        output_file = os.path.join(a.output_dir, converted_file_name)
                        audio = librosa.util.normalize(audio.astype(np.float32))
                        write(output_file, h.sampling_rate, audio)

                
if __name__ == '__main__':
    main()
