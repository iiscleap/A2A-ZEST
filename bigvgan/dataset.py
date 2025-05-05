# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import os
import pickle5 as pickle
import torch.nn.functional as F
import torchaudio

MAX_WAV_VALUE = 1


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 25.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 20.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_audio(full_path):
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def parse_manifest(manifest):
    audio_files = []
    codes = []
    rejected_files = ['p251_048_mic1.npy', 'p272_189_mic1.npy', 'p285_194_mic1.npy', 'p284_049_mic1.npy', 'p249_119_mic1.npy', 'p281_436_mic1.npy', 'p271_357_mic1.npy', 'p284_300_mic1.npy', 'p288_056_mic1.npy', 'p275_037_mic1.npy', 'p287_071_mic1.npy', 'p311_137_mic1.npy', 'p363_295_mic1.npy', 'p231_348_mic1.npy', 'p258_041_mic1.npy', 'p288_362_mic1.npy', 'p334_028_mic1.npy', 'p234_150_mic1.npy', 'p247_220_mic1.npy', 'p255_072_mic1.npy', 'p277_189_mic1.npy', 'p265_030_mic1.npy', 'p239_211_mic1.npy', 'p239_265_mic1.npy', 'p281_031_mic1.npy', 'p283_383_mic1.npy', 'p271_109_mic1.npy', 'p363_397_mic1.npy', 'p284_037_mic1.npy', 'p241_197_mic1.npy', 'p294_296_mic1.npy', 'p363_215_mic1.npy', 'p285_284_mic1.npy', 'p252_335_mic1.npy', 'p285_058_mic1.npy', 'p241_078_mic1.npy', 'p227_316_mic1.npy', 'p227_390_mic1.npy', 'p285_083_mic1.npy', 'p363_101_mic1.npy', 'p302_215_mic1.npy', 'p234_073_mic1.npy', 'p227_029_mic1.npy', 'p239_084_mic1.npy', 'p374_094_mic1.npy', 'p285_300_mic1.npy', 'p374_136_mic1.npy', 'p363_200_mic1.npy', 'p231_408_mic1.npy', 'p258_089_mic1.npy', 'p284_145_mic1.npy', 'p283_086_mic1.npy', 'p249_197_mic1.npy', 'p274_252_mic1.npy', 'p363_413_mic1.npy', 'p227_223_mic1.npy', 'p360_373_mic1.npy', 'p239_145_mic1.npy', 'p285_112_mic1.npy', 'p258_363_mic1.npy', 'p285_282_mic1.npy', 'p231_383_mic1.npy', 'p271_136_mic1.npy', 'p241_343_mic1.npy', 'p249_046_mic1.npy', 'p231_362_mic1.npy']
    rejected_files = [x.replace(".npy", ".wav") for x in rejected_files]
    #reqd_files = ["0011_000021", "0012_000022", "0013_000025", "0014_000032", "0015_000034", "0016_000035", "0017_000038", "0018_000043", "0019_000023", "0020_000047"]
    #for f in reqd_files[1:]:
    with open(manifest) as info:
        
        for line in info.readlines():
            # print("(dataset.py) line is ", line)
            # if f not in line:
            #     continue
            if line[0] == '{':
                sample = eval(line.strip())
                if sample["audio"].split(os.sep)[-1] in rejected_files:
                    continue
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                else:
                    k = 'hubert'

                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split('|')[0].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes


def get_dataset_filelist(h):
    training_files, training_codes = parse_manifest(h.input_training_file)
    print(len(training_files), len(training_codes))
    validation_files, validation_codes = parse_manifest(h.input_validation_file)

    return (training_files, training_codes), (validation_files, validation_codes)


def parse_speaker(path, method):
    if type(path) == str:
        path = Path(path)

    if method == 'parent_name':
        return path.parent.name
    elif method == 'parent_parent_name':
        return path.parent.parent.name
    elif method == '_':
        return path.name.split('_')[0]
    elif method == 'single':
        return 'A'
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


# def collate_fn(batch):
#     max_length = max(tensor.size(1) for tensor in batch)

#     #Pad tensors to the same length
#     padded_batch = [F.pad(tensor, (0, max_length - tensor.size(1)), "constant", 0) for tensor in batch]
#     # Stack the padded tensors
#     stacked_batch = torch.stack(padded_batch)
#     return stacked_batch


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, f0=None, energy=None, multispkr=False, pad=None,emo_embed=False,lang_embed=False,
                 f0_stats=None, f0_normalize=False, f0_feats=False, f0_median=False, energy_folder="",
                 f0_interp=False, vqvae=False, pitch_folder="", emo_folder="", spkr_folder="", lang_folder="", spkr_average=False):
        self.audio_files, self.codes = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.vqvae = vqvae
        self.f0 = f0
        self.energy = energy
        self.f0_normalize = f0_normalize
        self.f0_feats = f0_feats
        self.f0_stats = None
        self.f0_interp = f0_interp
        self.f0_median = f0_median
        self.pitch_folder = pitch_folder
        self.energy_folder = energy_folder
        self.emo_folder = emo_folder
        self.lang_folder = lang_folder
        self.spkr_folder = spkr_folder
        self.emo_embed = emo_embed
        self.lang_embed = lang_embed
        if f0_stats:
            self.f0_stats = torch.load(f0_stats)
        self.multispkr = multispkr
        self.pad = pad
        self.spkr_average = spkr_average
        if self.multispkr:
            spkrs = [parse_speaker(f, self.multispkr) for f in self.audio_files]
            spkrs = ["0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
            spkrs.sort()

            self.id_to_spkr = spkrs
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):

        filename = self.audio_files[index]
        # print(self.audio_files)
        # print("dataset filename line 232: ", filename)

        # filename = filename.replace(".wav", ".npy")
        # audio_info = sf.info(filename)
        # duration = audio_info.duration
        # if duration > 10:
        #     return self.__getitem__((index + 1) % len(self.audio_files))

       
        

        emo_file_name = str(filename).split(os.sep)[-1].replace(".wav", ".npy")
        # print("dataset emo_file_name line 235: ", emo_file_name)
        #if file path exists then load else set to None
        if not os.path.exists(os.path.join(self.pitch_folder, emo_file_name)):
            pitch = 0
        else:
            pitch = np.load(os.path.join(self.pitch_folder, emo_file_name))
            pitch = np.reshape(pitch, (1, -1))
        if not os.path.exists(os.path.join(self.energy_folder, emo_file_name)):
            energy = 0
        else:
            energy = np.load(os.path.join(self.energy_folder, emo_file_name))
            energy = np.reshape(energy, (1, -1))
            # pitch = np.log(pitch+1)
        # if not os.path.exists(os.path.join(self.spkr_folder, emo_file_name)):
        #     emo_embed = 0
        # else:
        #     emo_embed = np.load(os.path.join(self.spkr_folder, emo_file_name))

            # Check duration of audio file
            # audio_info = sf.info(filename)
            # duration = audio_info.duration
            # print(f"Duration of {filename}: {duration} seconds")
            
            # Skip processing if duration is more than 10 seconds
        
                
                
        if not os.path.exists(os.path.join(self.emo_folder, emo_file_name)):
            emo_embed = 0
        else:
            emo_embed = np.load(os.path.join(self.emo_folder, emo_file_name))
        if not os.path.exists(os.path.join(self.lang_folder, emo_file_name)):
            lang_embed = 0
        else:
            lang_embed = np.load(os.path.join(self.lang_folder, emo_file_name))


        # print(os.path.join(self.emo_folder, emo_file_name), emo_embed)
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                # raise ValueError("{} SR doesn't match target {} SR".format(
                #     sampling_rate, self.sampling_rate))
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Trim audio ending
        if self.vqvae:
            code_length = audio.shape[0] // self.code_hop_size
        else:
            code_length = min(audio.shape[0] // self.code_hop_size, self.codes[index].shape[0])
            code = self.codes[index][:code_length]
            if pitch.any() != 0:
                pitch = pitch[:, :code_length]
            else:
                pitch = np.zeros(1, code_length)
            if self.energy:
                if energy.any() != 0:
                    energy = energy[:, :code_length]
                else:
                    energy = np.zeros(1, code_length)
            # pitch = pitch[:code_length]
        audio = audio[:code_length * self.code_hop_size]

        assert self.vqvae or audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            if not self.vqvae:
                code = np.hstack([code, code])
                pitch = np.hstack([pitch, pitch])
                if self.energy:
                    energy = np.hstack([energy, energy])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        if self.vqvae:
            audio = self._sample_interval([audio])[0]
        else:
            if self.energy:
                audio, code, pitch, energy = self._sample_interval([audio, code, pitch, energy])
            else:
                audio, code, pitch = self._sample_interval([audio, code, pitch])
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        if self.vqvae:
            feats = {
                "code": audio.view(1, -1).numpy()
            }
        else:
            feats = {"code": code.squeeze()}

        if self.f0:
            f0 = torch.tensor(pitch)#.unsqueeze(0).unsqueeze(0)
            feats['f0'] = f0.numpy()
        if self.energy:
            energy = torch.tensor(energy)#.unsqueeze(0).unsqueeze(0)
            feats['energy'] = energy.numpy()
            # print(feats['f0'].shape, feats['code'].shape, filename)
        if self.multispkr:
            feats['spkr'] = np.load(os.path.join(self.spkr_folder, emo_file_name))

        if self.f0_normalize:
            # ii = feats['f0'] != 0
            feats['f0'] = np.log(feats['f0']+1)
            # mean = np.mean(feats['f0'][ii])
            # feats['f0'][ii] = (feats['f0'][ii] - mean)




        if self.spkr_average:
            with open('speakers.pkl', 'rb') as handle:
                speakers_feat = pickle.load(handle)
            feats['spkr'] = speakers_feat[emo_file_name[:4]]

            if self.f0_feats:
                feats['f0_stats'] = torch.FloatTensor([mean, std]).view(-1).numpy()
        if self.emo_embed:
            feats["emo_embed"] = emo_embed.squeeze(0)
        if self.lang_embed:
            feats["lang_embed"] = lang_embed.squeeze(0)
        # print(feats['f0'].shape, feats['energy'].shape)
        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def __len__(self):
        return len(self.audio_files)


