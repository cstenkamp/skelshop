## Environment variables
import os
from os.path import dirname
from os.path import join as pjoin


def cnf(name, val):
    globals()[name] = config.setdefault(name, val)

# Intermediate dirs
cnf("WORK", "work")
cnf("VIDEO_BASE", ".")
cnf("DUMP_BASE", ".")
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

def all(ext):
    base, = glob_wildcards(pjoin(VIDEO_BASE, "{base}.mp4"))
    return [fn + ext for fn in base]

rule sorted_all:
    input:
        [pjoin(DUMP_BASE, fn) for fn in all(".opt_lighttrack.h5")]
    output:
        touch(".sorted_all")

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
        pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        pjoin(DUMP_BASE, "{base}-Scenes.csv")
    run:
        workdir = dirname(wildcards.base)
        shell(
            "scenedetect --input {input} --output " + workdir +
            " detect-content --min-scene-len 2s list-scenes"
        )

rule skel_unsorted:
    input:
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        pjoin(DUMP_BASE, "{base}.unsorted.h5")
    shell:
        "python skelshop.py dump " +
        "--mode BODY_25_ALL " + 
        "{input.video} " + 
        "{output}"

rule skel_filter_csvshotseg_opt_lighttrack:
    input:
        gcn_config = GCN_CONFIG,
        unsorted = pjoin(DUMP_BASE, "{base}.unsorted.h5"),
        scenes_csv = pjoin(DUMP_BASE, "{base}-Scenes.csv")
    output:
        pjoin(DUMP_BASE, "{base}.opt_lighttrack.h5")
    shell:
        "python skelshop.py filter " +
        "--track " +
        "--track-conf opt_lighttrack " +
        "--pose-matcher-config {input.gcn_config} " +
        "--shot-seg=csv " +
        "--shot-csv {input.scenes_csv} " +
        "{input.unsorted} {output}"

rule drawsticks:
    input:
        skels = pjoin(DUMP_BASE, "{base}.{var}.h5"),
        video = pjoin(VIDEO_BASE, "{base}.mp4")
    output:
        pjoin(DUMP_BASE, "{base}.{var}.sticks.mp4")
    shell:
        "python skelshop.py drawsticks " +
        "{input.skels} {input.video} {output}"
