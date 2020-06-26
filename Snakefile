## Environment variables
import os
from os.path import dirname
from os.path import join as pjoin


def cnf(name, val):
    globals()[name] = config.setdefault(name, val)

# Intermediate dirs
cnf("WORK", "work")
cnf("GCN_WEIGHTS", WORK + "/gcn_weights")
cnf("GCN_CONFIG", WORK + "/gcn_config.yaml")

## Configs

GCN_INFERNENCE_YAML = """
weights: {gcn_weights}

# model
model: lighttrack.graph.gcn_utils.gcn_model.Model
model_args:
  in_channels: 2
  num_class: 128 # output feature dimension
  edge_importance_weighting: True
  graph_args:
    layout: 'PoseTrack'
    strategy: 'spatial'

# testing
device: [0]
""".strip()

## Rules

rule setup:
    input:
        GCN_CONFIG

rule vid_all:
    input:
        csvshotseg_reidpt = "{base}.reidpt.sticks.mp4",
        csvshotseg_reidman = "{base}.reidman.sticks.mp4"
    output:
        "{base}.all"
    shell:
        "touch {output}"

rule get_gcn_weights:
    output:
        directory(GCN_WEIGHTS)
    shell:
        "mkdir -p " + GCN_WEIGHTS + " && " +
        "cd " + GCN_WEIGHTS + " && " +
        "wget http://guanghan.info/download/Data/LightTrack/weights/GCN.zip && " + 
        "unzip GCN.zip"

rule tmpl_gcn_config:
    input:
        GCN_WEIGHTS
    output:
        GCN_CONFIG
    run:
        open(GCN_CONFIG, "w").write(
            GCN_INFERNENCE_YAML.format(
                gcn_weights=pjoin(os.getcwd(), GCN_WEIGHTS, "GCN/epoch210_model.pt")
            )
        )


rule scenedetect:
    input:
        "{base}.mp4"
    output:
        "{base}-Scenes.csv"
    run:
        workdir = dirname(wildcards.base)
        shell(
            "scenedetect --input {input} --output " + workdir +
            " detect-content --min-scene-len 2s list-scenes"
        )

rule skel_unsorted:
    input:
        video = "{base}.mp4"
    output:
        "{base}.unsorted.h5"
    shell:
        "python skelshop.py dump " +
        "--mode BODY_25_ALL " + 
        "{input.video} " + 
        "{output}"

rule skel_filter_csvshotseg_reidpt:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = "{base}.unsorted.h5",
        scenes_csv = "{base}-Scenes.csv"
    output:
        "{base}.reidpt.h5"
    shell:
        "python skelshop.py filter " +
        "--track " +
        "--track-reid-embed=posetrack " +
        "--pose-matcher-config {input.gcn_config} " +
        "--shot-seg=csv " +
        "--shot-csv {input.scenes_csv} " +
        "{input.unsorted} {output}"

rule skel_filter_csvshotseg_reidman:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = "{base}.unsorted.h5",
        scenes_csv = "{base}-Scenes.csv"
    output:
        "{base}.reidman.h5"
    shell:
        "python skelshop.py filter " +
        "--track " +
        "--track-reid-embed=manual " +
        "--pose-matcher-config {input.gcn_config} " +
        "--shot-seg=csv " +
        "--shot-csv {input.scenes_csv} " +
        "{input.unsorted} {output}"

rule drawsticks:
    input:
        skels = "{base}.{var}.h5",
        video = "{base}.mp4"
    output:
        "{base}.{var}.sticks.mp4"
    shell:
        "python skelshop.py drawsticks " +
        "{input.skels} {input.video} {output}"
