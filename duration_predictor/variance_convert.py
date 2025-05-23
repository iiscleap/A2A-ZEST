import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertModel
from transformers import Wav2Vec2Model
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import torch.nn.functional as F
from config import hparams
import pickle5 as pickle
import ast
from torch.autograd import Function
import argparse

parser = argparse.ArgumentParser(description="Pitch")
parser.add_argument(
    "--in_codes",
    metavar="in_codes",
    type=str,
)
parser.add_argument(
    "--spkr_folder",
    metavar="spkr_folder",
    type=str,
)
parser.add_argument(
    "--pitch_dir",
    metavar="pitch_dir",
    type=str,
)
parser.add_argument(
    "--tgt_emo_dir",
    metavar="tgt_emo_dir",
    type=str,
)
parser.add_argument(
    "--energy_dir",
    metavar="energy_dir",
    type=str,
)
parser.add_argument(
    "--target_pickle",
    metavar="target_pickle",
    type=str
)
parser.add_argument(
    "--output_file",
    metavar="output_file",
    type=str
)
args = parser.parse_args()


torch.set_printoptions(profile="full")
#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class WAV2VECModel(nn.Module):
    def __init__(self,
                 wav2vec,
                 output_dim,
                 hidden_dim_emo):
        
        super().__init__()
        
        self.wav2vec = wav2vec
        
        embedding_dim = wav2vec.config.to_dict()['hidden_size']
        self.out = nn.Linear(hidden_dim_emo, output_dim)
        self.out_spkr = nn.Linear(hidden_dim_emo, 10)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        
    def forward(self, aud, alpha):
        aud = aud.squeeze(0)
        hidden_all = list(self.wav2vec(aud).hidden_states)
        embedded = sum(hidden_all)
        embedded = embedded.permute(0, 2, 1)

        emo_embedded = self.relu(self.conv1(embedded))
        emo_embedded = self.relu(self.conv2(emo_embedded))
        emo_embedded = emo_embedded.permute(0, 2, 1)
        emo_hidden = torch.mean(emo_embedded, 1).squeeze(1)

        out_emo = self.out(emo_hidden)

        reverse_feature = ReverseLayerF.apply(embedded, alpha)

        embedded_spkr = self.relu(self.conv3(reverse_feature))
        embedded_spkr = self.relu(self.conv4(embedded_spkr))
        hidden_spkr = torch.mean(embedded_spkr, -1).squeeze(-1)
        output_spkr = self.out_spkr(hidden_spkr)
        
        return out_emo, output_spkr, emo_hidden, emo_embedded

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim_q, hidden_dim_k):
        super().__init__()
        HIDDEN_SIZE = 256
        NUM_ATTENTION_HEADS = 4
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        self.fc_q = nn.Linear(hidden_dim_q, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)

        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.5,
                                                    bias = True,
                                                    batch_first=True)
                                                                                                           
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_q)
        self.fc_1 = nn.Linear(hidden_dim_q, hidden_dim_q)
        self.relu = nn.ReLU()
    
    def forward(self, query_i, key_i, value_i):
        query = self.fc_q(query_i)
        key = self.fc_k(key_i)
        value = self.fc_v(value_i)
        cross, _ = self.multihead_attn(query, key, value, need_weights = True)
        skip = self.fc(cross)
 
        skip += query_i
        skip = self.relu(skip)
        skip = self.layer_norm(skip)

        new = self.fc_1(skip)
        new += skip
        new = self.relu(new)
        out = self.layer_norm_1(new)
        
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Positional Encoding (optional, can be learned)
        self.positional_encoding = nn.Parameter(torch.randn(1, feature_dim, 1))  # Learnable positional encoding
        
        # Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.positional_encoding[:,:,:x.size(-1)]  # Match the sequence length
        x = self.transformer_encoder(x.permute(2, 0, 1))  # Shape: (L, 32, 128)
        return x.permute(1, 2, 0) 

class PitchModel(nn.Module):
    def __init__(self, hparams):
        super(PitchModel, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.wav2vec = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)
        self.encoder = WAV2VECModel(self.wav2vec, 5, hparams["emotion_embedding_dim"])
        self.embedding = nn.Embedding(101, 128, padding_idx=100)        
        self.fusion = CrossAttentionModel(128, 128)
        self.fusion_dur = CrossAttentionModel(128, 128)
        self.linear_layer = nn.Linear(128, 1)
        self.leaky = nn.LeakyReLU()
        self.cnn_reg1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.cnn_reg2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.cnn_reg3 = nn.Conv1d(256, 256, kernel_size=(3,), padding=1)
        self.cnn_reg4 = nn.Conv1d(256, 256, kernel_size=(3,), padding=1)
        self.cnn_reg5 = nn.Conv1d(256, 256, kernel_size=(3,), padding=1)
        self.cnn_reg6 = nn.Conv1d(256, 256, kernel_size=(3,), padding=1)
        self.ln_1 = nn.LayerNorm(256)
        self.ln_2 = nn.LayerNorm(256)
        self.ln_3 = nn.LayerNorm(256)
        self.ln_4 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.5)
        self.dur_linear = nn.Linear(256, 1)

    def process_duration(self, code, code_feat):
        uniq_code_count = []
        uniq_code_feat = []
        for i in range(code.size(0)):
            uniq_code, count = torch.unique_consecutive(code[i, :], return_counts=True)
            if len(count) > 2:
                # print(count)
                # remove first and last code as segment sampling may cause incomplete segment length
                uniq_code_count.append(count[:-1])
                uniq_code_idx = count.cumsum(dim=0) - 1
            else:
                uniq_code_count.append(count)
                uniq_code_idx = count.cumsum(dim=0) - 1
            uniq_code_idx = uniq_code_idx[:-1]
            uniq_code_feat.append(code_feat[i, uniq_code_idx, :].view(-1, code_feat.size(2)))
        uniq_code_count = torch.cat(uniq_code_count)
        # collate feat
        max_len = max(feat.size(0) for feat in uniq_code_feat)
        out = uniq_code_feat[0].new_zeros((len(uniq_code_feat), max_len, uniq_code_feat[0].size(1)))
        mask = torch.arange(max_len).repeat(len(uniq_code_feat), 1)
        for i, v in enumerate(uniq_code_feat):
            out[i, : v.size(0)] = v
            mask[i, :] = mask[i, :] < v.size(0)

        return out, mask.bool(), uniq_code_count.float(), uniq_code[:-1]

    def get_emo(self, aud):
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, _ = self.encoder(inputs['input_values'].to(device), 1.0)

        return emo_hidden
    
    def inference_nodur(self, aud, tokens, speaker):
        hidden = self.embedding(tokens.int())
        # uniq_code_feat = hidden
        # uniq_code = tokens.int()
        # uniq_code_feat, _, _, uniq_code = self.process_duration(tokens, hidden)
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, emo_embedded = self.encoder(inputs['input_values'].to(device), 1.0)
        speaker_temp = speaker.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1)
        emo_embedded = emo_embedded + speaker_temp
        
        fusion = self.fusion(hidden, emo_embedded, emo_embedded)
        fusion = fusion.permute(0, 2, 1)
        pred_pitch = self.cnn_reg2(self.leaky(self.cnn_reg1(fusion)))
        pred_pitch = pred_pitch.squeeze(1)
        
        
        return tokens, pred_pitch, emo_hidden

    def inference(self, aud, tokens, speaker):
        hidden = self.embedding(tokens.int())
        uniq_code_feat, _, _, uniq_code = self.process_duration(tokens, hidden)
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, emo_embedded = self.encoder(inputs['input_values'].to(device), 1.0)
        speaker_temp = speaker.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1)
        emo_embedded = emo_embedded + speaker_temp


        emo_hidden_repeat = emo_hidden.unsqueeze(1).repeat(1, uniq_code_feat.shape[1], 1)
        speaker_1 = speaker.unsqueeze(1).repeat(1, uniq_code_feat.shape[1], 1)
        combined = torch.cat((uniq_code_feat, emo_hidden_repeat), -1)
        combined = combined.permute(0, 2, 1)
        pred_dur = self.leaky(self.cnn_reg3(combined))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.dropout(self.ln_1(pred_dur))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.leaky(self.cnn_reg4(pred_dur))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.dropout(self.ln_2(pred_dur))

        combined = torch.cat((uniq_code_feat, speaker_1), -1)
        combined = combined.permute(0, 2, 1)
        pred_dur_1 = self.leaky(self.cnn_reg5(combined))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.dropout(self.ln_3(pred_dur_1))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.leaky(self.cnn_reg6(pred_dur_1))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.dropout(self.ln_4(pred_dur_1))

        pred_dur += pred_dur_1
        
        pred_dur = self.dur_linear(pred_dur)
        pred_dur = pred_dur.squeeze(-1)
        pred_dur = torch.clamp(torch.round(torch.exp(pred_dur)-1).long(), min=1)
        tokens_dup = torch.repeat_interleave(uniq_code, pred_dur.view(-1))
        hidden = self.embedding(tokens_dup.int()).unsqueeze(0)
        
        fusion = self.fusion(hidden, emo_embedded, emo_embedded)
        fusion = fusion.permute(0, 2, 1)
        pred_pitch = self.cnn_reg2(self.leaky(self.cnn_reg1(fusion)))
        pred_pitch = pred_pitch.squeeze(1)
        # pred_energy = self.cnn_reg4(self.leaky(self.cnn_reg3(fusion)))
        # pred_energy = pred_energy.squeeze(1)

        
        return tokens_dup, pred_pitch, emo_hidden


    def forward(self, aud, tokens, speaker, lengths, alpha=1.0):
        hidden = self.embedding(tokens.int())
        uniq_code_feat, uniq_code_mask, dur, uniq_code = self.process_duration(tokens, hidden)
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        emo_out, spkr_out, emo_hidden, emo_embedded = self.encoder(inputs['input_values'].to(device), alpha)
        speaker_temp = speaker.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1)
        emo_embedded = emo_embedded + speaker_temp
        fusion = self.fusion(hidden, emo_embedded, emo_embedded)
        fusion = fusion.permute(0, 2, 1)

        pred_pitch = self.cnn_reg2(self.leaky(self.cnn_reg1(fusion)))
        pred_pitch = pred_pitch.squeeze(1)
        
        emo_hidden_repeat = emo_hidden.unsqueeze(1).repeat(1, uniq_code_feat.shape[1], 1)
        speaker_1 = speaker.unsqueeze(1).repeat(1, uniq_code_feat.shape[1], 1)
        combined = torch.cat((uniq_code_feat, emo_hidden_repeat), -1)
        combined = combined.permute(0, 2, 1)
        pred_dur = self.leaky(self.cnn_reg3(combined))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.dropout(self.ln_1(pred_dur))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.leaky(self.cnn_reg4(pred_dur))
        pred_dur = pred_dur.permute(0, 2, 1)
        pred_dur = self.dropout(self.ln_2(pred_dur))

        combined = torch.cat((uniq_code_feat, speaker_1), -1)
        combined = combined.permute(0, 2, 1)
        pred_dur_1 = self.leaky(self.cnn_reg5(combined))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.dropout(self.ln_3(pred_dur_1))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.leaky(self.cnn_reg6(pred_dur_1))
        pred_dur_1 = pred_dur_1.permute(0, 2, 1)
        pred_dur_1 = self.dropout(self.ln_4(pred_dur_1))

        pred_dur += pred_dur_1
        
        pred_dur = self.dur_linear(pred_dur)
        pred_dur = pred_dur.squeeze(-1)
        pred_dur = pred_dur[uniq_code_mask]
        mask = torch.arange(hidden.shape[1]).expand(hidden.shape[0], hidden.shape[1]).to(device) < lengths.unsqueeze(1)
        pred_pitch = pred_pitch.masked_fill(~mask, 0.0)
        mask = mask.int()

        return pred_pitch, pred_dur, emo_out, spkr_out, mask, dur, uniq_code_mask

def get_codes(in_file):
    codes = {}
    with open(in_file) as f:
        lines = f.readlines()
        for l in lines:
            d = ast.literal_eval(l)
            name, tokens = d["audio"], d["hubert"].split(" ")
            codes[name] = np.array(tokens).astype(int)
    return codes


def get_f0_dict():
    
    
    model = PitchModel(hparams)
    model.to(device)
    model.load_state_dict(torch.load('pitch_duration_predictor.pth'))
    
    model.eval()

    codes = get_codes(args.in_codes)
    os.makedirs(args.pitch_dir, exist_ok=True)
    os.makedirs(args.tgt_emo_dir, exist_ok=True)
    # os.makedirs(args.src_emo_dir, exist_ok=True)
    targets = json.load(open(args.target_pickle, "r"))
    samples = []
    
    with torch.no_grad():
        for src_file, target_files in targets.items():
            speaker_path = os.path.join(args.spkr_folder, src_file.split(os.sep)[-1].replace(".wav", ".npy"))
            speaker = torch.tensor(np.load(speaker_path)).unsqueeze(0).to(device)
            tokens = codes[src_file]
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            tokens = torch.cat((tokens, 100*torch.ones((1,1)).to(device)), -1)
            src_audio = torchaudio.load(src_file)[0].to(device)
            if src_audio.shape[0] == 2:
                src_audio = torch.mean(src_audio, 0).unsqueeze(0)
            for target in target_files:
                sample = {}
                inputs = torchaudio.load(target)[0].to(device)
                if inputs.shape[0] == 2:
                    inputs = torch.mean(inputs, 0).unsqueeze(0)
                
                tokens_dup, pitch_pred, tgt_emo_embedding = model.inference(inputs, tokens, speaker)
                tokens_dup = tokens_dup.squeeze(0).cpu().detach().numpy()
                tokens_dup = list(tokens_dup)
                tokens_dup = [str(int(x)) for x in tokens_dup]
                tokens_dup = " ".join(tokens_dup)
                samples += [sample]
                pitch_pred = torch.exp(pitch_pred) - 1
                final_name = src_file.split(os.sep)[-1].replace(".wav", "") + "--" + target.split(os.sep)[-1]
                sample["audio"] = final_name
                sample["hubert"] = tokens_dup
                final_name = final_name.replace(".wav", ".npy")
                np.save(os.path.join(args.pitch_dir, final_name), pitch_pred[0, :].cpu().detach().numpy())
                np.save(os.path.join(args.tgt_emo_dir, final_name), tgt_emo_embedding[0, :].cpu().detach().numpy())
        out_path = os.path.join(args.output_file)
        with open(out_path, 'w') as f:
            f.write('\n'.join([str(x) for x in samples]))

def get_f0_only():
    
    
    model = PitchModel(hparams)
    model.to(device)
    model.load_state_dict(torch.load('pitch_duration_predictor.pth'))
    
    model.eval()

    codes = get_codes(args.in_codes)
    os.makedirs(args.pitch_dir, exist_ok=True)
    os.makedirs(args.tgt_emo_dir, exist_ok=True)
    # os.makedirs(args.src_emo_dir, exist_ok=True)
    targets = json.load(open(args.target_pickle, "r"))
    samples = []
    
    with torch.no_grad():
        for src_file, target_files in targets.items():
            speaker_path = os.path.join(args.spkr_folder, src_file.split(os.sep)[-1].replace(".wav", ".npy"))
            speaker = torch.tensor(np.load(speaker_path)).unsqueeze(0).to(device)
            for target in target_files:
                final_name = src_file.split(os.sep)[-1].replace(".wav", "") + "--" + target.split(os.sep)[-1]
                tokens = codes[final_name]
                tokens = torch.tensor(tokens).unsqueeze(0).to(device)
                inputs = torchaudio.load(target)[0].to(device)
                if inputs.shape[0] == 2:
                    inputs = torch.mean(inputs, 0).unsqueeze(0)
                
                _, pitch_pred, tgt_emo_embedding = model.inference_nodur(inputs, tokens, speaker)
                pitch_pred = torch.exp(pitch_pred) - 1
                final_name = final_name.replace(".wav", ".npy")
                np.save(os.path.join(args.pitch_dir, final_name), pitch_pred[0, :].cpu().detach().numpy())
                np.save(os.path.join(args.tgt_emo_dir, final_name), tgt_emo_embedding[0, :].cpu().detach().numpy())
        # out_path = os.path.join(args.output_file)
        # with open(out_path, 'w') as f:
        #     f.write('\n'.join([str(x) for x in samples]))
                

if __name__ == "__main__":
    get_f0_only()
