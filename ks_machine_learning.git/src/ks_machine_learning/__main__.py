import sys
import click

from ks_machine_learning.cli.ra_data_info import ra_data_info
from ks_machine_learning.cli.ra_data_nl import ra_data_nl
from ks_machine_learning.cli.plotting import plotting

@click.group()
def cli():
    pass

cli.add_command(ra_data_info)
cli.add_command(ra_data_nl)
cli.add_command(plotting)
