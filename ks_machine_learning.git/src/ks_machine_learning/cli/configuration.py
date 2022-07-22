import logging
import click
import configparser
import ast
from os.path import isfile, join, dirname, realpath, abspath
from ks_machine_learning import __package_dir__ as PKG_DIR,\
                               __def_cfg_dir__ as CFG_DIR,\
                               __user_home_dir__ as HOME_DIR,\
                               __raw_data_dir__ as RAW_DATA_DIR,\
                               __output_dir__ as OUTPUT_DIR,\
                               __csv_dir__ as CSV_DIR,\
                               __nl_data_dir__ as NL_DATA_DIR,\
                               __ts_data_dir__ as TS_DATA_DIR,\
                               __fig_dir__ as FIG_DIR
from .utils import add_options

log = logging.getLogger(__name__)

DESCRIPTION_EXTRACT = 'Listing the config of recording audio signal'

DATA_INFO_OPTIONS = [
    click.option(
        "-c",
        "--config",
        type=click.Path(readable=True, dir_okay=False),
        help="config",
        default=join(PKG_DIR,
                     CFG_DIR,
                     "ra_data_info.conf"),
        show_default=True
    ),
    click.option(
        "-rd",
        "--data-info-raw-data-dir",
        type=click.Path(dir_okay=True),
        help="directory of raw data",
        default=join(PKG_DIR,
                     RAW_DATA_DIR),
        show_default=True
    ),
    click.option(
        "-l",
        "--data-info-list",
        is_flag=True,
        help="list data info",
    ),
    click.option(
        "-o",
        "--data-info-output-csv",
        type=click.Path(exists=True, dir_okay=True),
        help="output csv file",
        default=join(PKG_DIR,
                     OUTPUT_DIR,
                     CSV_DIR),
    ),
]

DATA_NL_OPTIONS = [
    click.option(
        "-c",
        "--config",
        type=click.Path(readable=True, dir_okay=False),
        help="config",
        default=join(PKG_DIR,
                     CFG_DIR,
                     "ra_data_nl.conf"),
        show_default=True
    ),
    click.option(
        "-rd",
        "--data-nl-raw-data-dir",
        type=click.Path(dir_okay=True),
        help="directory of raw data",
        default=join(PKG_DIR,
                     RAW_DATA_DIR),
        show_default=True
    ),
    click.option(
        "-cnd",
        "--data-nl-ch-nl-data-dir",
        type=click.Path(dir_okay=True),
        help="directory of channel normalization data",
        default=join(PKG_DIR,
                     OUTPUT_DIR,
                     NL_DATA_DIR),
        show_default=True
    ),
    click.option(
        "-tnd",
        "--data-nl-ts-nl-data-dir",
        type=click.Path(dir_okay=True),
        help="directory of temporal normalization data",
        default=join(PKG_DIR,
                     OUTPUT_DIR,
                     TS_DATA_DIR),
        show_default=True
    ),
    click.option(
        "-chsp",
        "--data-nl-channel-separation",
        is_flag=True,
        help="channel separation of raw data",
    ),
    click.option(
        "-ts",
        "--data-nl-temporal-split",
        is_flag=True,
        help="temporal split of raw data",
    ),
]

PLOTTING_OPTIONS = [
    click.option(
        "-c",
        "--config",
        type=click.Path(readable=True, dir_okay=False),
        help="config",
        default=join(PKG_DIR,
                     CFG_DIR,
                     "plotting.conf"),
        show_default=True
    ),
    click.option(
        "-d",
        "--plotting-data-dir",
        type=click.Path(dir_okay=True),
        help="directory of raw data",
        default=join(PKG_DIR,
                     RAW_DATA_DIR),
        show_default=True
    ),
    click.option(
        "-o",
        "--plotting-output-dir",
        type=click.Path(exists=True, dir_okay=True),
        help="output fig file",
        default=join(PKG_DIR,
                     OUTPUT_DIR,
                     FIG_DIR),
    ),
    click.option(
        "-sp",
        "--plotting-spectrogram",
        is_flag=True,
        help="plotting spectrogram",
    ),
    click.option(
        "-mt",
        "--plotting-magnitude-spectrum",
        is_flag=True,
        help="plotting spectrogram",
    ),
]

class Configuration:

    def __init__(
            self,
            config=None,
            data_info=False,
            data_info_raw_data_dir=None,
            data_info_list=None,
            data_info_output_csv=None,
            data_nl=False,
            data_nl_raw_data_dir=None,
            data_nl_ch_nl_data_dir=None,
            data_nl_channel_separation=False,
            data_nl_ts_nl_data_dir=None,
            data_nl_temporal_split=False,
            plotting=False,
            plotting_data_dir=None,
            plotting_output_dir=None,
            plotting_spectrogram=False,
            plotting_magnitude_spectrum=False,
    ):
        self.config=config
        self.cfg_dict={}

        self.data_info=data_info
        self.data_nl=data_nl
        self.plotting=plotting

        if self.data_info:
            self.data_info_raw_data_dir=data_info_raw_data_dir
            self.data_info_list=data_info_list
            self.data_info_output_csv=data_info_output_csv

        if self.data_nl:
            self.data_nl_raw_data_dir=data_nl_raw_data_dir
            self.data_nl_ch_nl_data_dir=data_nl_ch_nl_data_dir
            self.data_nl_channel_separation=data_nl_channel_separation
            self.data_nl_ts_nl_data_dir=data_nl_ts_nl_data_dir
            self.data_nl_temporal_split=data_nl_temporal_split

        if self.plotting:
            self.plotting_data_dir=plotting_data_dir
            self.plotting_output_dir=plotting_output_dir
            self.plotting_spectrogram=plotting_spectrogram
            self.plotting_magnitude_spectrum=plotting_magnitude_spectrum

        self._load_config()

    def _load_config(self):
        conf_parser = configparser.ConfigParser()

        if isfile(self.config):
            log.info(f"Found config file {self.config}")
            conf_parser.read(self.config)
            main_conf = conf_parser['main']

            for key in main_conf:
                setattr(self, key, ast.literal_eval(main_conf[key.upper()]))
                self.cfg_dict[f'{key}'] = main_conf[key.upper()]
        else:
            log.info(f"config file {self.config} not found")
