# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

'''
python train.py --checkpoint_path models_nounitenc/ --config hifi_pitch_nounitenc.json --spkr_folder /data1/soumyad/ZEST_updated/EASE/ease_vectors --pitch_folder /data1/soumyad/multilingual_ZEST/pitch_predictor/pitch --emo_folder /data1/soumyad/ZEST_updated/pitch_predictor/emo_embed
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from bigvgan import CodeGenerator
from discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)

from utils import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_audio,
    build_env,
    AttrDict
)

torch.backends.cudnn.benchmark = True


def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))

    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator(h).to(device)
    mrd = MultiResolutionDiscriminator(h).to(device)

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None),energy=h.get('energy', None), multispkr=h.get('multispkr', None),energy_folder=a.energy_folder,
                           f0_stats=h.get('f0_stats', None), emo_embed=h.get('emo_embed', None), lang_embed=h.get('lang_embed', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False), pitch_folder=a.pitch_folder, spkr_folder = a.spkr_folder,
                           lang_folder=a.lang_folder, emo_folder=a.emo_folder, spkr_average=h.get('spkr_average', None))

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=0, shuffle=False, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                               fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),energy=h.get('energy', None),
                               multispkr=h.get('multispkr', None), lang_folder=a.lang_folder,
                               f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                               f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False), energy_folder=a.energy_folder,
                               f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False), pitch_folder=a.pitch_folder,
                               emo_folder=a.emo_folder,spkr_folder = a.spkr_folder, spkr_average=h.get('spkr_average', None))
        validation_loader = DataLoader(validset, num_workers=0, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
            y_g_hat = generator(**x)
            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                y_g_hat, commit_losses, metrics = y_g_hat

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            if h.get('f0_vq_params', None):
                f0_commit_loss = commit_losses[1][0]
                f0_metrics = metrics[1][0]
            if h.get('code_vq_params', None):
                code_commit_loss = commit_losses[0][0]
                code_metrics = metrics[0][0]

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f
            clip_grad_norm = h.get("clip_grad_norm", 1000.0)  # Default to 1000

            # Whether to freeze D for initial training steps
            if steps >= a.freeze_step:
                loss_disc_all.backward()
                grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                    mpd.parameters(), clip_grad_norm
                )
                grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                    mrd.parameters(), clip_grad_norm
                )
                optim_d.step()
            else:
                print(
                    f"[WARNING] skipping D training for the first {a.freeze_step} steps"
                )
                grad_norm_mpd = 0.0
                grad_norm_mrd = 0.0

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            lambda_melloss = h.get(
                "lambda_melloss", 45.0
            )  # Defaults to 45 in BigVGAN-v1 if not set
            if h.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                loss_mel = fn_mel_loss_multiscale(y, y_g_hat) * lambda_melloss
            else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * lambda_melloss

            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            if steps >= a.freeze_step:
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )
            else:
                print(
                    f"[WARNING] using regression loss only for G for the first {a.freeze_step} steps"
                )
                loss_gen_all = loss_mel

            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), clip_grad_norm
            )
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to stdout
                    print(
                        f"Steps: {steps:d}, "
                        f"Gen Loss Total: {loss_gen_all:4.3f}, "
                        f"Mel Error: {mel_error:4.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                        f"grad_norm_g: {grad_norm_g:4.3f}"
                    )

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                # if steps % a.summary_interval == 0:
                #     sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                #     sw.add_scalar("training/mel_spec_error", mel_error, steps)
                #     if h.get('f0_vq_params', None):
                #         sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                #         sw.add_scalar("training/used_curr", f0_metrics['used_curr'].item(), steps)
                #         sw.add_scalar("training/entropy", f0_metrics['entropy'].item(), steps)
                #         sw.add_scalar("training/usage", f0_metrics['usage'].item(), steps)
                #     if h.get('code_vq_params', None):
                #         sw.add_scalar("training/code_commit_error", code_commit_loss, steps)
                #         sw.add_scalar("training/code_used_curr", code_metrics['used_curr'].item(), steps)
                #         sw.add_scalar("training/code_entropy", code_metrics['entropy'].item(), steps)
                #         sw.add_scalar("training/code_usage", code_metrics['usage'].item(), steps)

                # Validation
                # if steps % a.validation_interval == 0:  # and steps != 0:
                #     generator.eval()
                #     torch.cuda.empty_cache()
                #     val_err_tot = 0
                #     with torch.no_grad():
                #         for j, batch in enumerate(validation_loader):
                #             x, y, _, y_mel = batch
                #             x = {k: v.to(device, non_blocking=False) for k, v in x.items()}

                #             y_g_hat = generator(**x)
                #             if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                #                 y_g_hat, commit_losses, _ = y_g_hat

                #             if h.get('f0_vq_params', None):
                #                 f0_commit_loss = commit_losses[1][0]
                #                 val_err_tot += f0_commit_loss * h.get('lambda_commit', None)

                #             if h.get('code_vq_params', None):
                #                 code_commit_loss = commit_losses[0][0]
                #                 val_err_tot += code_commit_loss * h.get('lambda_commit_code', None)
                #             y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                #             y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                #                                           h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                #             val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                #             if j <= 4:
                #                 if steps == 0:
                #                     sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                #                     sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                #                 sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                #                 y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                #                                              h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                #                 sw.add_figure('generated/y_hat_spec_{}'.format(j),
                #                               plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                #         val_err = val_err_tot / (j + 1)
                #         sw.add_scalar("validation/mel_spec_error", val_err, steps)
                #         if h.get('f0_vq_params', None):
                #             sw.add_scalar("validation/commit_error", f0_commit_loss, steps)
                #         if h.get('code_vq_params', None):
                #             sw.add_scalar("validation/code_commit_error", code_commit_loss, steps)
                #     generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--pitch_folder', default='')
    parser.add_argument('--energy_folder', default='')
    parser.add_argument('--spkr_folder', default='')
    parser.add_argument('--emo_folder', default='')
    parser.add_argument('--lang_folder', default='')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=15000000000, type=int)
    parser.add_argument('--stdout_interval', default=1000, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=10000000000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed-world-size', type=int)
    parser.add_argument('--distributed-port', type=int)
    parser.add_argument(
        "--freeze_step",
        default=0,
        type=int,
        help="freeze D for the first specified steps. G only uses regression loss for these steps.",
    )


    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print('Batch size per GPU :', h.batch_size)
    else:
        rank = 0
        local_rank = 0

    train(rank, local_rank, a, h)


if __name__ == '__main__':
    main()
