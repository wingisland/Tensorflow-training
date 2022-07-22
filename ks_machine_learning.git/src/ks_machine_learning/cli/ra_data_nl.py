import logging
import click
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from os import listdir, path
from collections import defaultdict
from ks_machine_learning.cli.configuration import Configuration,\
                                                  DATA_NL_OPTIONS
from ks_machine_learning.tools.common_utility import listdir_fullpath,\
                                                     listdir_endis_ls_hidden,\
                                                     WavFileNormalization
from .utils import add_options

log = logging.getLogger(__name__)

DESCRIPTION_EXTRACT = 'raw audio data normalization'

@click.command(help=DESCRIPTION_EXTRACT, name="ranl")
@add_options(DATA_NL_OPTIONS)
def ra_data_nl(**kwargs):
    #
    config=Configuration(data_nl=True, **kwargs)
    wfnl=WavFileNormalization(config)
    half_num_of_cpu=round(cpu_count()/2)
    pool=Pool(half_num_of_cpu)
    
    if config.data_nl_channel_separation:
        result=pool.map(wfnl.wav_channel_separation,
                        listdir_fullpath(config.data_nl_raw_data_dir))

    if config.data_nl_temporal_split:
        result=pool.map(wfnl.wav_temporal_split,
                        listdir_fullpath(config.data_nl_ch_nl_data_dir))
