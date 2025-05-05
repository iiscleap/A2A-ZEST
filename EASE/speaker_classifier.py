import os
import torch
import torchaudio
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import torch.nn.functional as F
import ast
from torch.autograd import Function
import pandas as pd

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

class MyDataset(Dataset):
    '''Dataset class for audio which reads in the audio signals and prepares them for
    training. In particular it pads all of them to the maximum length of the audio
    signal that is present. All audio signals are padded/sampled upto 34s in this
    dataset.
    '''
    def __init__(self, speaker_folder, files, labels):
        self.speaker_folder = speaker_folder
        self.files = files
        self.label_dict = {"neutral":0, "angry":1, "happy":2, "sad":3, "surprise":4}
        self.labels_list = labels
        self.sr = 16000
        self.speaker_dict = {}
        for ind in range(11, 21):
            self.speaker_dict["00"+str(ind)] = ind-11

    def __len__(self):
        return len(self.files)

    def getspkrlabel(self, file_name):
        spkr_name = file_name[:4]
        spkr_label = self.speaker_dict[spkr_name]

        return spkr_label

        
    def __getitem__(self, audio_ind):
        speaker_feat = np.load(os.path.join(self.speaker_folder, self.files[audio_ind].replace(".wav", ".npy")))
        speaker_label = self.getspkrlabel(self.files[audio_ind])
        class_id = self.label_dict[self.labels_list[audio_ind]] 

        return speaker_feat, speaker_label, class_id, self.files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class SpeakerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(192, 128)
        self.fc = nn.Linear(128, 128)
        self.fc_embed = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_embed_1 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(128, 128)
        self.fc_embed_2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 5)
    
    def forward(self, feat, alpha=1.0):
        feat = self.fc(self.fc_embed(self.fc1(feat)))
        reverse = ReverseLayerF.apply(feat, alpha)
        out = self.fc3(self.fc_embed_1(self.fc2(feat)))
        emo_out = self.fc5(self.fc_embed_2(self.fc4(reverse)))
        
        return out, emo_out, feat

def create_dataset(mode, bs=32):
    speaker_folder = "/target/path/for/x-vectors/"
    df_metadata = pd.read_csv("metadata.csv")
    if mode == 'train':
        df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="train")]
        files = df_metadata["path"].values
        files = [x.split(os.sep)[-1].replace(".wav", ".npy") for x in files]
        labels = df_metadata["emotion"].values
    elif mode == 'val':
        df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="valid")]
        files = df_metadata["path"].values
        files = [x.split(os.sep)[-1].replace(".wav", ".npy") for x in files]
        labels = df_metadata["emotion"].values
    elif mode =="test":
        df_metadata = df_metadata[(df_metadata["language"]=="english") & (df_metadata["split"]=="test")]
        files = df_metadata["path"].values
        files = [x.split(os.sep)[-1].replace(".wav", ".npy") for x in files]
        labels = df_metadata["emotion"].values
    else:
        folder = "/home/soumyad/TIMIT/wav_files"
        speaker_folder = "/home/soumyad/Librispeech/speaker_embeddings"
        token_file = "/home/soumyad/Librispeech/librispeech_parsed.txt"
    dataset = MyDataset(speaker_folder, files, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)

    return loader

def train():
    
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = SpeakerModel()
    model.to(device)
    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params':parameters, 'lr':base_lr}])
    final_val_loss = 1e20

    for e in range(10):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs, out_emo, _ = model(speaker_feat, alpha)
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            loss_emo = nn.CrossEntropyLoss(reduction='mean')(out_emo, emo_labels)
            loss += loss_emo
            tot_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pred = torch.argmax(outputs, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs, out_emo, _ = model(speaker_feat)
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
                val_loss += loss.detach().item()
                pred = torch.argmax(outputs, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save(model, 'ease.pth')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = accuracy_score(gt_tr, pred_tr)
        val_loss_log = val_loss/len(val_loader)
        val_f1 = accuracy_score(gt_val, pred_val)
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")

def get_ease_embedding():
    model = torch.load('ease.pth', map_location=device)
    model.to(device)
    model.eval()
    folder = "/path/to/store/ease/vectors/"
    xvector_folder = "/path/to/x-vectors/"
    speaker_files = os.listdir(xvector_folder)
    speaker_files = [x for x in speaker_files if ".npy" in x]
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(tqdm(speaker_files)):
            target_file_name = os.path.join(folder, data)
            if os.path.exists(target_file_name):
                continue
            speaker_feat = torch.tensor(np.load(os.path.join(xvector_folder, data))).to(device)
            _, _, embedded = model(speaker_feat)
            target_file_name = os.path.join(folder, data)
            np.save(os.path.join(folder, target_file_name), embedded.cpu().detach().numpy())

if __name__ == "__main__":
    # train()
    get_ease_embedding()
