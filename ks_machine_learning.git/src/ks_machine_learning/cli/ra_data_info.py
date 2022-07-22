import logging
import click
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from os import listdir, path
from collections import defaultdict
from ks_machine_learning.cli.configuration import Configuration,\
                                                  DATA_INFO_OPTIONS
from ks_machine_learning.tools.common_utility import AudioProcess,\
                                                     listdir_fullpath,\
                                                     listdir_endis_ls_hidden
from .utils import add_options

log = logging.getLogger(__name__)

DESCRIPTION_EXTRACT = 'raw audio data information'

@click.command(help=DESCRIPTION_EXTRACT, name="radi")
@add_options(DATA_INFO_OPTIONS)
def ra_data_info(**kwargs):
    #
    config=Configuration(data_info=True, **kwargs)
    ap=AudioProcess(num_of_file=len(listdir_endis_ls_hidden(
                                        config.data_info_raw_data_dir)))
    audio_raw_data_dict=defaultdict(list)
    half_num_of_cpu=round(cpu_count()/2)
    pool=Pool(half_num_of_cpu)

    fn_list=listdir_endis_ls_hidden(config.data_info_raw_data_dir)
    fs_list=pool.imap(ap.get_audio_attr_fs,\
                      listdir_fullpath(config.data_info_raw_data_dir))
    ch_list=pool.imap(ap.get_audio_attr_ch,\
                      listdir_fullpath(config.data_info_raw_data_dir))
    len_list=pool.imap(ap.get_audio_attr_len,\
                       listdir_fullpath(config.data_info_raw_data_dir))

    audio_raw_data_dict.update({f"{config.file_name_title}": fn_list})
    audio_raw_data_dict.update({f"{config.audio_sample_rate_title}": fs_list})
    audio_raw_data_dict.update({f"{config.audio_channel_title}": ch_list})
    audio_raw_data_dict.update({f"{config.audio_length_title}": len_list})

    raw_data_df = pd.DataFrame(audio_raw_data_dict)
   
    raw_data_df.to_csv(path.join(config.data_info_output_csv,
                                 config.csv_file_name),
                       index=False)

    if config.data_info_list:
        print(raw_data_df)
