import logging
import wave
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import  make_chunks
from os import listdir, path
from pathlib import Path
from tqdm import tqdm
from scipy import signal

log = logging.getLogger(__name__)

class AudioProcess:

    def __init__(
            self,
            num_of_file=None,
    ):
        self.num_of_file=num_of_file
        print(f"the total number of files : {self.num_of_file}")
    def get_audio_attr_fs(self, file_path):
        file_extension=Path(file_path).suffix.replace('.', '')
        audio_file=AudioSegment.from_file(file=file_path, format=file_extension)
        return audio_file.frame_rate
    
    def get_audio_attr_ch(self, file_path):
        file_extension=Path(file_path).suffix.replace('.', '')
        audio_file=AudioSegment.from_file(file=file_path, format=file_extension)
        return audio_file.channels
    
    def get_audio_attr_len(self, file_path):
        file_extension=Path(file_path).suffix.replace('.', '')
        audio_file=AudioSegment.from_file(file=file_path, format=file_extension)
        return len(audio_file)

class WavFileNormalization:

    def __init__(
            self,
            config,
    ):
        self.config=config
        self.num_of_readframes=config.num_of_readframes
        self.num_of_channel_for_sp=config.num_of_channel_for_sp
        self.depth_type_dict={ 1: np.int8,2: np.int16, 4: np.int32 }
        self.data_nl_ch_nl_data_dir=config.data_nl_ch_nl_data_dir
        self.chunk_length_ms=config.chunk_length_ms
        self.format_of_audio_file=config.format_of_audio_file
        self.num_of_zero_padding_for_fn=config.num_of_zero_padding_for_fn
        self.data_nl_ts_nl_data_dir=config.data_nl_ts_nl_data_dir

    def wav_channel_separation(self, wav_fp):
        wav=wave.open(wav_fp, 'r')
        channel=wav.getnchannels()
        frames=wav.readframes(self.num_of_readframes)
        depth=wav.getsampwidth()
        frame_rate=wav.getframerate()
        data_type_per_depth=self.depth_type_dict.get(depth)
        if not data_type_per_depth:
            raise ValueError(f"sample width {depth} not supported")

        sdata_per_frame=np.fromstring(frames, dtype=data_type_per_depth)

        if channel > self.num_of_channel_for_sp:
            for ch_idx in range(channel):
                _audio_file_output_per_chsp(sdata_per_frame[ch_idx::channel],
                                            wav.getparams(),
                                            self.num_of_channel_for_sp,
                                            ch_idx,
                                            wav_fp,
                                            self.data_nl_ch_nl_data_dir)

    def wav_temporal_split(self, wav_fp):
        audio_file=AudioSegment.from_file(wav_fp, self.format_of_audio_file)
        chunks=make_chunks(audio_file, self.chunk_length_ms)
        _audio_file_output_per_ts(chunks,
                                  wav_fp,
                                  self.format_of_audio_file,
                                  self.num_of_zero_padding_for_fn,
                                  self.data_nl_ts_nl_data_dir)

class SpectrogramPlotting:

    def __init__(
            self,
            config,
    ):
        self.config=config
        self.fig_width=self.config.fig_width
        self.fig_heigth=self.config.fig_heigth
        self.fig_subplot=self.config.fig_subplot
        self.num_of_readframes=self.config.num_of_readframes
        self.data_type_of_frames=self.config.data_type_of_frames
        self.fig_color_map=self.config.fig_color_map
        self.fig_specgram_nfft=self.config.fig_specgram_nfft
        self.fig_specgram_noverlap=self.fig_specgram_nfft/2
        self.plotting_output_dir=self.config.plotting_output_dir
        self.num_channel_of_mono=self.config.num_channel_of_mono
        self.depth_type_dict={ 1: np.int8,2: np.int16, 4: np.int32 }
        self.num_of_max_power_freq=self.config.num_of_max_power_freq
        self.freq_of_ultrasonic=self.config.freq_of_ultrasonic
        self.status_label_of_unknown=self.config.status_label_of_unknown
        self.status_label_of_working=self.config.status_label_of_working
        self.status_label_of_idle=self.config.status_label_of_idle
        self.freq_of_ultrasonic=self.config.freq_of_ultrasonic

    def graph_spectrogram(self, file_name, pdir):
        sdata_per_frame, frame_rate, channel = _get_wav_info(path.join(pdir,
                                                                       file_name),
                                                             self.num_of_readframes,
                                                             self.depth_type_dict)
        if channel == self.num_channel_of_mono:
            fig_file_path=path.join(path.join(self.plotting_output_dir, file_name))
            plt.figure(figsize=(self.fig_width, self.fig_heigth))
            plt.subplot(self.fig_subplot)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.specgram(sdata_per_frame,
                         Fs=frame_rate,
                         NFFT=self.fig_specgram_nfft,
                         noverlap=self.fig_specgram_noverlap,
                         cmap=self.fig_color_map)
            plt.savefig(fig_file_path.replace("wav", "png"))
            plt.close()
        else:
            pass

    def graph_magnitude(self, file_name, pdir):
        sdata_per_frame, frame_rate, channel = _get_wav_info(path.join(pdir,
                                                                       file_name),
                                                             self.num_of_readframes,
                                                             self.depth_type_dict)
        if channel == self.num_channel_of_mono:
            fig_file_path=path.join(path.join(self.plotting_output_dir, file_name))
            freq, Pxx_den=signal.welch(sdata_per_frame, frame_rate, 'flattop', self.fig_specgram_nfft, scaling='spectrum')
            plt.semilogy(freq, Pxx_den)
            plt.ylim([1, 10e10])
            plt.stem([4359.375, 4781.25, 5156.25, 5953.125, 7546.875, 8343.75, 8765.625, 10453.125], [10e10, 10e10, 10e10, 10e10, 10e10, 10e10, 10e10, 10e10,], linefmt='r--')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Linear spectrum [V RMS]')
            plt.savefig(fig_file_path.replace("wav", "png"))
            plt.close()
        else:
            pass

    def magnitude_info(self, file_name, pdir):
        sdata_per_frame, frame_rate, channel = _get_wav_info(path.join(pdir,
                                                                       file_name),
                                                             self.num_of_readframes,
                                                             self.depth_type_dict)
        max_mt_freq=None
        if channel == self.num_channel_of_mono:
            fig_file_path=path.join(path.join(self.plotting_output_dir, file_name))
            freq, Pxx_den=signal.welch(sdata_per_frame, frame_rate, 'flattop', self.fig_specgram_nfft, scaling='spectrum')
            max_mt_freq=freq[np.where(Pxx_den == np.amax(Pxx_den))].tolist()
        else:
            pass

        return max_mt_freq

    def __status_of_mpf__(self, mpf_list):
        __status__="unknown"
        if len(mpf_list) == self.num_of_max_power_freq:
            if mpf_list[0] >=  self.freq_of_ultrasonic:
                __status__=self.status_label_of_working
            else:
                __status__=self.status_label_of_idle
        else:
            __status__=self.status_label_of_unknown

        return __status__

    def __len_of_mpf__(self, mpf_list):
        return len(mpf_list)

def listdir_fullpath(d, endis_ls_hidden=False):
    if endis_ls_hidden:
        return sorted(path.join(d, f) for f in listdir(d))
    else:
        return sorted(path.join(d, f) for f in listdir(d) if not f.startswith("."))

def listdir_endis_ls_hidden(d, endis_ls_hidden=False):
    if endis_ls_hidden:
        return sorted(f for f in listdir(d))
    else:
        return sorted(f for f in listdir(d) if not f.startswith("."))

def _audio_file_output_per_chsp(sdata, audio_params, nch_for_sp, ch_idx, file_path, pdir):
    PARENT_DIR=pdir
    OUTPUT_FILE_PATH=path.join(PARENT_DIR, path.basename(file_path))
    outwav=wave.open(OUTPUT_FILE_PATH.replace(".wav", f"_ch{ch_idx}.wav"), 'w')
    outwav.setparams(audio_params)
    outwav.setnchannels(nch_for_sp)
    outwav.writeframes(sdata.tostring())
    outwav.close()
    
def _audio_file_output_per_ts(chunks, file_path, file_format, num_zp, pdir):
    PARENT_DIR=pdir
    OUTPUT_FILE_PATH=path.join(PARENT_DIR, path.basename(file_path))
    for i, chunk in enumerate(chunks):
        chunk_name = OUTPUT_FILE_PATH.replace(f".{file_format}", f"_ck_{i:{num_zp}}.{file_format}")
        chunk.export(f"{chunk_name}", format=file_format)

def _get_wav_info(wav_file, num_of_readframes, depth_type_dict):
    wav=wave.open(wav_file, 'r')
    frames=wav.readframes(num_of_readframes)
    channel=wav.getnchannels()
    depth=wav.getsampwidth()
    frame_rate=wav.getframerate()
    data_type_per_depth=depth_type_dict.get(depth)
    if not data_type_per_depth:
        raise ValueError(f"sample width {depth} not supported")

    sdata_per_frame=np.fromstring(frames, dtype=data_type_per_depth)
    wav.close()
    return sdata_per_frame, frame_rate, channel
