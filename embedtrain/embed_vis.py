import os
import h5py
import click
import numpy
import cv2
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot as plt
from sklearn import manifold
from time import time
from os.path import join as pjoin
from skeldump.embed.manual import angle_embed_pose_joints
from embedtrain.datasets import HandDataSet, BodyDataSet
from ordered_set import OrderedSet
from embedtrain.draw import draw
from embedtrain.utils import resize_sq_aspect
from embedtrain.embed_skels import EMBED_SKELS


def mk_manual_embeddings(skel, h5f):
    print("Making manual embeddings")
    result = []

    def proc_item(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        result.append((name, angle_embed_pose_joints(skel, obj)))
    h5f.visititems(proc_item)
    print("Done making manual embeddings")
    return result


@click.group()
def embed_vis():
    pass


SPRITE_SIZE = 32


@embed_vis.command()
@click.argument("h5fin")
@click.argument("log_dir")
@click.argument("skel_name")
@click.option(
    "--image-base", envvar="IMAGE_BASE", type=click.Path(exists=True)
)
@click.option(
    "--body-labels", envvar="BODY_LABELS", type=click.Path(exists=True)
)
def to_tensorboard(h5fin, log_dir, skel_name, image_base, body_labels):
    if skel_name != "HAND":
        assert body_labels
    skel = EMBED_SKELS[skel_name]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorboard.plugins import projector
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    embeddings = []
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as metadataf, h5py.File(h5fin, "r") as h5f:
        if skel_name == "HAND":
            metadataf.write("name\tsrc\tcls\tsrc_cls\n")
        else:
            metadataf.write("name\tact_id\tcat_name\tact_name\n")
        manual_embeddings = mk_manual_embeddings(skel, h5f)
        print("Writing embeddings and metadata")
        if image_base is not None:
            print("...and sprite sheet")
            sprite_sheet = numpy.empty((len(manual_embeddings) * SPRITE_SIZE, SPRITE_SIZE, 3))
        for idx, (path, embedding) in enumerate(manual_embeddings):
            if skel_name == "HAND":
                src, cls = HandDataSet.path_to_dataset_class_pair(path)
                metadataf.write(f"{path}\t{src}\t{cls}\t{src}-{cls}\n")
            else:
                cls = BodyDataSet.path_to_class(body_labels, path)
                metadataf.write(f"{path}\t{cls['act_id']}\t{cls['cat_name']}\t{cls['act_name']}\n")
            embeddings.append(embedding)
            if image_base is not None:
                im = draw(h5f, image_base, path, skel)
                sprite_sheet[idx * SPRITE_SIZE:(idx + 1) * SPRITE_SIZE, :] = \
                        resize_sq_aspect(im, SPRITE_SIZE)
        if image_base is not None:
            cv2.imwrite(pjoin(log_dir, "sprites.png", sprite_sheet))
        print("Done writing embeddings and metadata")

    weights = tf.Variable(embeddings)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_config.metadata_path = 'metadata.tsv'
    if image_base is not None:
        embedding_config.sprite.image_path = 'sprites.png'
        embedding_config.sprite.single_image_dim.extend([SPRITE_SIZE, SPRITE_SIZE])
    print("Performing dimensionality reduction")
    projector.visualize_embeddings(log_dir, config)
    print("Done performing dimensionality reduction")

    print("Now run: tensorboard serve --logdir " + log_dir)


def embeddings_by_class(h5fin):
    classes = OrderedSet()
    x = []
    y = []
    for path, embedding in mk_manual_embeddings(h5fin):
        src_cls = HandDataSet.path_to_dataset_class_pair(path)
        cls_idx = classes.add(src_cls)
        x.append(embedding)
        y.append(cls_idx)
    return x, y


def nulltight(ax):
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


@embed_vis.command()
@click.argument("h5fin")
@click.argument("outdir")
def tsne_multi(h5fin, outdir):
    x, y = embeddings_by_class(h5fin)

    n_components = 2
    (fig, subplots) = plt.subplots(3, figsize=(15, 8))
    perplexities = [5, 30, 50]

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i]

        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        x2 = tsne.fit_transform(x)
        t1 = time()
        print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(x2[:, 0], x2[:, 1], c=y)
        nulltight(ax)
    plt.show()


if __name__ == "__main__":
    embed_vis()