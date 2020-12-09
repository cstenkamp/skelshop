from pathlib import Path
from subprocess import Popen
from typing import TextIO

import click

from skelshop.utils.click import PathPath

DEFAULT_THUMBVIEW_CMD = "sxiv -t"


@click.command()
@click.argument("protos_dir", type=PathPath())
@click.argument("assign_out", type=click.File("w"))
@click.option("--thumbview-cmd", default=DEFAULT_THUMBVIEW_CMD)
def whoisthis(protos_dir: Path, assign_out: TextIO, thumbview_cmd: str):
    assign_out.write("label,clus\n")
    for clus_dir in protos_dir.iterdir():
        thumbs_proc = Popen(thumbview_cmd.split(" ") + [str(clus_dir)])
        answer = input("Who is this? (type 'quit' when finished) ")
        thumbs_proc.kill()
        thumbs_proc.wait()
        if answer == "quit":
            break
        assign_out.write("{},{}\n".format(answer, clus_dir.name))
