Bootstrap: docker
From: frankierr/skelshop:latest

%runscript
    cd /opt/skelshop && snakemake "$@"

%environment
    source /.openpose_env

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export LD_LIBRARY_PATH=$OPENPOSE/src/openpose/:$LD_LIBRARY_PATH
    export MODEL_FOLDER=$OPENPOSE_MODELS
