import argparse
import time
import sys
import pickle

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

from sklearn.random_projection import gaussian_random_matrix

import numpy as np
import pandas as pd

print('TF version: {}'.format(tf.__version__))
print('TF-Hub version: {}'.format(hub.__version__))

# Globals
METRIC = 'angular'

embed_fn = None
projected_dim = 64

def print_with_time(msg):
    print('{}: {}'.format(time.ctime(), msg))
    sys.stdout.flush()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sentences')
    parser.add_argument('-use_model', default='https://tfhub.dev/google/universal-sentence-encoder/4', type=str)
    parser.add_argument('-csv_file_path', default='./short-wiki.csv', type=str)
    parser.add_argument('-ann', default='./wiki.annoy.index', type=str)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-num_trees', default=10, type=int)
    parser.add_argument('-random_projection', default=0, type=int, help='1 to ON, and 0 to OFF')
    return parser.parse_args()


def read_data(path):
  df_docs = pd.read_csv(path, usecols=['GUID', 'CONTENT'])
  df_docs.head()
  return df_docs.to_numpy()

def generate_random_projection_weights(original_dim, projected_dim):
  random_projection_matrix = None
  random_projection_matrix = gaussian_random_matrix(
      n_components=projected_dim, n_features=original_dim).T
  print("A Gaussian random weight matrix was creates with shape of {}".format(random_projection_matrix.shape))
  print('Storing random projection matrix to disk...')
  with open('random_projection_matrix', 'wb') as handle:
    pickle.dump(random_projection_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
  return random_projection_matrix

def generate_embeddings(text, module_url, random_projection_matrix=None):
  # Beam will run this function in different processes that need to
  # import hub and load embed_fn (if not previously loaded)
  global embed_fn
  if embed_fn is None:
    embed_fn = hub.load(module_url)
  embedding = embed_fn(text).numpy()
  if random_projection_matrix is not None:
    embedding = embedding.dot(random_projection_matrix)

  return embedding

def build_index(batch_size, content_array, model_url, random_projection_matrix):
    VECTOR_LENGTH = 512

    if random_projection_matrix is not None:
      VECTOR_LENGTH = 64

    ann = AnnoyIndex(VECTOR_LENGTH, metric=METRIC)

    batch_sentences = []
    batch_indexes = []
    last_indexed = 0
    num_batches = 0

    for sindex, sentence in enumerate(content_array):
      # sentence_embedding = generate_embeddings(sentence[1], model_url, random_projection_matrix)
      # ann.add_item(sindex, sentence_embedding[0])

      batch_sentences.append(sentence[1])
      batch_indexes.append(sindex)

      if len(batch_sentences) == batch_size:
        context_embed = generate_embeddings(batch_sentences, model_url, random_projection_matrix)
        for index in batch_indexes:
          ann.add_item(index, context_embed[index - last_indexed])
          batch_sentences = []
          batch_indexes = []
        last_indexed += batch_size
        if num_batches % 10000 == 0:
          print_with_time('sindex: {} annoy_size: {}'.format(sindex, ann.get_n_items()))
        num_batches += 1

    if batch_sentences:
      context_embed = generate_embeddings(batch_sentences, model_url, random_projection_matrix)
      for index in batch_indexes:
        ann.add_item(index, context_embed[index - last_indexed])

    return ann

def main():
    args = setup_args()
    print_with_time(args)

    start_time = time.time()
    content_array = read_data(args.csv_file_path)
    end_time = time.time()
    print('Read Data Time: {}'.format(end_time - start_time))

    random_projection_matrix = None

    if args.random_projection:
      if projected_dim:
        original_dim = hub.load(args.use_model)(['']).shape[1]
        random_projection_matrix = generate_random_projection_weights(original_dim, projected_dim)

    start_time = time.time()
    ann = build_index(args.batch_size, content_array, args.use_model, random_projection_matrix)
    end_time = time.time()
    print('Build Index Time: {}'.format(end_time - start_time))

    ann.build(args.num_trees)
    ann.save(args.ann)

if __name__ == '__main__':
    main()