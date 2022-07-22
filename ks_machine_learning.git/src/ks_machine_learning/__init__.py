import logging
import sys
from os.path import dirname, realpath
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
__package_dir__         = dirname(realpath(__file__))
__def_cfg_dir__         = 'def_config'
__user_home_dir__       = str(Path.home())
__raw_data_dir__        = 'raw_data'
__output_dir__          = 'output'
__csv_dir__             = 'csv'
__nl_data_dir__         = 'chsp_data'
__ts_data_dir__         = 'ts_data'
__fig_dir__             = 'fig'
