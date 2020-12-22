EMBED_SIZE = 128
DEFAULT_FRAME_BATCH_SIZE = 16
DEFAULT_FACES_BATCH_SIZE = 64

# XXX: Euclidean is for dlib embeddings, but other embeddings might be trained
# with cosine, similarly this detection threshold is only reasonable for dlib
DEFAULT_METRIC = "euclidean"
DEFAULT_DETECTION_THRESHOLD = 0.6
