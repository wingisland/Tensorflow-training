import logging
import click
import glob
import pandas as pd
import numpy
import sys
from multiprocessing import Pool, cpu_count
from os import listdir, path
from collections import defaultdict
from functools import partial
from ks_machine_learning.cli.configuration import Configuration,\
                                                  DATA_INFO_OPTIONS,\
                                                  PLOTTING_OPTIONS
from ks_machine_learning.tools.common_utility import SpectrogramPlotting,\
                                                     listdir_endis_ls_hidden
from .utils import add_options

log = logging.getLogger(__name__)

DESCRIPTION_EXTRACT = 'plotting'

@click.command(help=DESCRIPTION_EXTRACT, name="plot")
@add_options(PLOTTING_OPTIONS)
def plotting(**kwargs):
    #
    config=Configuration(plotting=True, **kwargs)
    sp=SpectrogramPlotting(config)
    mpf_data_dict=defaultdict(list)
    half_num_of_cpu=round(cpu_count()/2)
    pool=Pool(half_num_of_cpu)

    fn_list=listdir_endis_ls_hidden(config.plotting_data_dir)

    if config.plotting_spectrogram:
        plt_sp_func=partial(sp.graph_spectrogram,
                            pdir=config.plotting_data_dir)
        pool.map(plt_sp_func, listdir_endis_ls_hidden(config.plotting_data_dir))

    if config.plotting_magnitude_spectrum:
        plt_sp_func=partial(sp.graph_magnitude,
                            pdir=config.plotting_data_dir)
        pool.map(plt_sp_func, listdir_endis_ls_hidden(config.plotting_data_dir))

    '''
    mt_info_func=partial(sp.magnitude_info,
                         pdir=config.plotting_data_dir)
    max_mtf_list=pool.map(mt_info_func, listdir_endis_ls_hidden(config.plotting_data_dir))
    num_of_mpf_list=pool.map(sp.__len_of_mpf__, max_mtf_list)
    dev_status_per_mpf=pool.map(sp.__status_of_mpf__, max_mtf_list)


    mpf_data_dict.update({f"file name": fn_list})
    mpf_data_dict.update({f"num of max power frequency": num_of_mpf_list})
    mpf_data_dict.update({f"max power frequency": max_mtf_list})
    mpf_data_dict.update({f"device status": dev_status_per_mpf})
    mpf_data_df=pd.DataFrame(mpf_data_dict)

    mpf_data_df.to_csv(path.join(config.plotting_output_dir,
                                 "device_status.csv"),
                       index=False)

    plt_sp_func=partial(sp.graph_spectrogram,
                        pdir=config.plotting_data_dir)
    plt_psd_func=partial(sp.graph_psd,
                         pdir=config.plotting_data_dir)
    psd_info_func=partial(sp.psd_info,
                          pdir=config.plotting_data_dir)

    mpf_list=pool.map(psd_info_func, listdir_endis_ls_hidden(config.plotting_data_dir))
    num_of_mpf_list=pool.map(sp.__len_of_mpf__, mpf_list)
    dev_status_per_mpf=pool.map(sp.__status_of_mpf__, mpf_list)

    pool.map(plt_sp_func, listdir_endis_ls_hidden(config.plotting_data_dir))
    # pool.map(plotting_func, listdir_endis_ls_hidden(config.plotting_data_dir))

    mpf_data_dict.update({f"file name": fn_list})
    mpf_data_dict.update({f"max power frequency": mpf_list})
    mpf_data_dict.update({f"num of max power frequency": num_of_mpf_list})
    mpf_data_dict.update({f"device status": dev_status_per_mpf})

    mpf_data_df = pd.DataFrame(mpf_data_dict)

    mpf_data_df.to_csv(path.join(config.plotting_data_dir,
                                 "hi.csv"),
                       index=False)

    print(mpf_data_df)
    '''
