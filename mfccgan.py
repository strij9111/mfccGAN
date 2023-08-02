import argparse
import math
import random
import time
from pathlib import Path

import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import numpy as np
import scipy.io.wavfile

from pystoi import stoi
from audiomentations import (
    Compose,
    RoomSimulator,
    AirAbsorption,
    TanhDistortion,
    TimeStretch,
    PitchShift,
    AddGaussianNoise,
    Gain,
    Shift,
    PolarityInversion
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import weight_norm
import torchaudio
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml


MRH_Dict = dict({
    'save_path': 'chkpt',
    'load_path': None,
    'n_mel_channels': 36,
    'ngf': 32,
    'n_residual_layers': 3,
    'ndf': 16,
    'num_D': 3,
    'n_layers_D': 4,
    'downsamp_factor': 4,
    'lambda_feat': 10,

    'data_path': 'wavs/',
    'batch_size': 256,
    'seq_len': 8000,
    'epochs': 1200,
    'log_interval': 100,
    'save_interval': 1000,
    'n_test_samples': 1
})

main_sample_rate = 32000
data_path = "data\\wavs"
main_hop_length = 64

save_path = MRH_Dict['chkpt']
load_path = MRH_Dict['load_path']
n_mel_channels = MRH_Dict['n_mel_channels']
ngf = MRH_Dict['ngf']
n_residual_layers = MRH_Dict['n_residual_layers']
ndf = MRH_Dict['ndf']
num_D = MRH_Dict['num_D']
n_layers_D = MRH_Dict['n_layers_D']
downsamp_factor = MRH_Dict['downsamp_factor']
lambda_feat = MRH_Dict['lambda_feat']
batch_size = MRH_Dict['batch_size']
seq_len = MRH_Dict['seq_len']
epochs = MRH_Dict['epochs']
log_interval = MRH_Dict['log_interval']
save_interval = MRH_Dict['save_interval']
n_test_samples = MRH_Dict['n_test_samples']


def pad_to_seq_len(x, seq_len):
    # Получаем текущую длину последовательности
    curr_seq_len = x.shape[-1]

    # Вычисляем, сколько нулей нужно добавить
    padding = seq_len - curr_seq_len

    if padding > 0:
        # Если нужно добавить нули, используем F.pad
        x = F.pad(x, (0, padding))
    elif padding < 0:
        # Если текущая длина последовательности больше seq_len, обрезаем до seq_len
        x = x[..., :seq_len]

    return x


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


filename = Path(data_path) / "training.txt"
files = files_to_list(filename)

audio_files = [filename.parent / x for x in files]

random.seed(2201)
random.shuffle(audio_files)


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]

        random.shuffle(self.audio_files)

        self.augment = augment
        if self.augment:
            self.augmentations = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.1),
                PitchShift(min_semitones=-6, max_semitones=6, p=0.1),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.1),
                Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.1),
                #                BandStopFilter(min_bandwidth_fraction=0.01, max_bandwidth_fraction=0.25, p=0.1),
                PolarityInversion(p=0.1),
                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.1),
                AirAbsorption(p=0.4),
                TanhDistortion(p=0.1),
                #                SevenBandParametricEQ(p=0.3)
            ])

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment

        if audio.shape[1] > self.segment_length:
            # Если длина аудио больше segment_length, выбираем произвольный отрывок
            start = torch.randint(0, audio.shape[1] - self.segment_length, (1,)).item()
            return audio[:, start: start + self.segment_length]
        else:
            # Если длина аудио меньше segment_length, дополняем нулями
            padding = torch.zeros((audio.shape[0], self.segment_length - audio.shape[1])).to(audio.device)
            return torch.cat((audio, padding), dim=1)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sample_rate = torchaudio.load(full_path)

        if self.augment:
            data = self.augmentations(samples=data.numpy(), sample_rate=self.sampling_rate)
            data = torch.from_numpy(data)

        return data, sample_rate


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=main_hop_length,
            win_length=1024,
            sampling_rate=main_sample_rate,
            n_mel_channels=36,
            mel_fmin=0.0,
            mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


"""
Генератор MFCCGAN: стек транспонированных сверточных слоев используется для увеличения
входного MFCC с низким временным разрешением до сырой речевой волны. За каждым сверточным слоем располагается блок остаточных
слоев с атрибутом дилатации. Дилатация приводит к большему рецептивному полю входного сигнала, что лучше моделирует вариации
Увеличение разрешения применяется на четырех этапах: два увеличения разрешения 8x и два увеличения 2x. 
На каждом этапе применяется оператор транспонированной свертки 1D для увеличения разрешения. 
Остаточные стеки не меняют размер входного сигнала. Каждый остаточный стек состоит из трех сверточных слоев с различными параметрами дилатации.
"""


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]
        MRH_in = 32
        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]
            kernel_size = r * 2
            stride = r
            padding = r // 2 + r % 2
            output_padding = r % 2
            MRH_dilation = 1
            MRH_length = (MRH_in - 1) * stride - 2 * padding + MRH_dilation * (kernel_size - 1) + output_padding + 1
            MRH_in = MRH_length

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = x.squeeze()
        return self.model(x)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice='conv', act='prelu'):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


"""
Дискриминатор имеет многоуровневую архитектуру, содержащую три идентичных дискриминатора,
которые работают на разных аудиошкалах (частотах дискретизации).
Понижение разрешения сигнала до более низкого разрешения позволяет рассматривать сигнал с различными частотными разрешениями.
Поведение более высокой частоты больше проявляется на более высоких шкалах, в то время как более низкая частота дискретизации
может отражать поведение низкой частоты. Таким образом, масштабирование выходного сигнала на три уровня может привести к лучшему суждению о качестве.
"""


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 32000)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


class Audio2MFCC(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=main_hop_length,
            win_length=1024,
            sampling_rate=main_sample_rate,
            n_mel_channels=36
    ):
        super().__init__()

        ##############################################
        # FFT Parameters                              #
        ##############################################
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

        # MFCC transformer
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sampling_rate,
            n_mfcc=self.n_mel_channels,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
            }
        )

    def forward(self, x):
        mfccs = self.mfcc_transform(x.cuda())
        return mfccs


def main():
    root = Path(save_path)
    load_root = Path(load_path) if load_path else None
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(MRH_Dict, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(n_mel_channels, ngf, n_residual_layers).cuda()
    netD = Discriminator(num_D, ndf, n_layers_D, downsamp_factor).cuda()

    myMFCC = Audio2MFCC(n_mel_channels=n_mel_channels, hop_length=main_hop_length).cuda()

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    schedulerG = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(data_path) / "training.txt", seq_len, sampling_rate=main_sample_rate
    )
    test_set = AudioDataset(
        Path(data_path) / "validation.txt",
        seq_len,
        sampling_rate=main_sample_rate,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=1)

    print('##################')

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        s_t = myMFCC(x_t).detach()
        s_t = s_t.cuda()
        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), main_sample_rate, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=main_sample_rate)

        if i == n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn auto tuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0

    print("\nTraining")
    for epoch in range(1, epochs + 1):
        print('Epoch ', epoch)

        for iterno, x_t in enumerate(train_loader):
            s_t = myMFCC(x_t).detach()
            x_pred_t = netG(s_t)

            with torch.no_grad():
                x_pred_t = pad_to_seq_len(x_pred_t, seq_len)
                s_pred_t = myMFCC(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################

            scaler = 10.0 / np.log(10.0) * np.sqrt(2)
            distortion = s_t - s_pred_t

            mcd = distortion.pow(2.0).sum(dim=-1).sqrt().mean(dim=-1) * scaler

            x1 = mcd.detach()

            mcd_mean = x1.mean()
            mcd_max = 1000
            mcd_score = abs(mcd_max - mcd_mean) / mcd_max

            orig = x_t.detach()
            pred = x_pred_t.detach()
            batch_num = orig.shape[0]

            orig_sig = torch.flatten(x_t)
            pred_sig = torch.flatten(pred)

            mystoi = stoi(orig_sig.detach().cpu().numpy().astype('float32'),
                          pred_sig.detach().cpu().numpy().astype('float32'), main_sample_rate,
                          extended=False)

            D_fake_det = netD(x_pred_t.detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += ((mystoi - scale[-1]) ** 2).mean()

            for scale in D_real:
                loss_D += ((1 - scale[-1]) ** 2).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda())

            loss_G = 0
            for scale in D_fake:
                loss_G += ((1 - scale[-1]) ** 2).mean()

            loss_feat = 0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            wt = D_weights * feat_weights
            for i in range(num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + lambda_feat * loss_feat).backward()
            optG.step()
            schedulerG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), main_sample_rate, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=main_sample_rate,
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("\n **Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)
            mystoi1 = round(mystoi, 3)
            if steps % log_interval == 0:
                print("mystoi:", mystoi1, "mcd_score:", mcd_score.item(),
                      "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                          epoch,
                          iterno,
                          len(train_loader),
                          1000 * (time.time() - start) / log_interval,
                          np.asarray(costs).mean(0),
                      )
                      )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()