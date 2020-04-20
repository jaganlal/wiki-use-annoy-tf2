import argparse
import time
import sys
import os
import pickle

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

import numpy as np
import pandas as pd
import csv

print('TF version: {}'.format(tf.__version__))
print('TF-Hub version: {}'.format(hub.__version__))

# Globals
D = 512
embed_fn = None

def print_with_time(msg):
  print('{}: {}'.format(time.ctime(), msg))
  sys.stdout.flush()


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-sentences')
  parser.add_argument('-use_model', default='https://tfhub.dev/google/universal-sentence-encoder/4', type=str)
  parser.add_argument('-csv_file_path', default='./short-wiki.csv', type=str)
  parser.add_argument('-ann', default='./wiki.annoy.index', type=str)
  parser.add_argument('-filter_data', default=0, type=int, help='1 to on, 0 to off')
  parser.add_argument('-k', default=10, type=int, help='# of neighbors')
  parser.add_argument('-vector_size', default=512, type=int, help='Annoy Index vector size')
  parser.add_argument('-random_projection', default=0, type=int, help='1 to ON, and 0 to OFF')
  return parser.parse_args()


def read_data(path, filter_data):
  df_docs = pd.read_csv(path, usecols=['GUID', 'CONTENT', 'ENTITY'])
  return df_docs

def extract_embeddings(query, embed_fn, random_projection_matrix):
  '''Generates the embedding for the query'''
  query_embedding =  embed_fn([query]).numpy()
  if random_projection_matrix is not None:
    query_embedding = query_embedding.dot(random_projection_matrix)
  return query_embedding


def generate_embeddings(text, module_url, random_projection_matrix=None):
  global embed_fn
  if embed_fn is None:
    embed_fn = hub.load(module_url)
  embedding = embed_fn([text])[0].numpy()
  if random_projection_matrix is not None:
    embedding = embedding.dot(random_projection_matrix)
  return embedding


def find_similar_items(ann, embedding, content_array, num_matches):
  '''Finds similar items to a given embedding in the ANN index'''
  ids = ann.get_nns_by_vector(
  embedding, num_matches, search_k=-1, include_distances=False)
  items = [content_array[i] for i in ids]
  return items

def main():
    args = setup_args()
    print_with_time(args)

    start_time = time.time()
    ann = AnnoyIndex(args.vector_size, metric='angular')
    ann.load(args.ann)
    end_time = time.time()
    print('Load Time: {}'.format(end_time - start_time))

    print_with_time('Annoy Index: {}'.format(ann.get_n_items()))

    start_time = time.time()
    df = read_data(args.csv_file_path, args.filter_data)
    content_array = df.to_numpy()
    end_time = time.time()
    print_with_time('Sentences: {} Time: {}'.format(len(content_array), end_time - start_time))

    # start_time = time.time()
    # embed_fn = hub.load(args.use_model)
    # end_time = time.time()
    # print_with_time('Model loaded time: {}'.format(end_time - start_time))

    random_projection_matrix = None

    if args.random_projection:
      if os.path.exists('random_projection_matrix'):
        print("Loading random projection matrix...")
        with open('random_projection_matrix', 'rb') as handle:
          random_projection_matrix = pickle.load(handle)
        print('random projection matrix is loaded.')

    while True:
      input_sentence_id = input('Enter sentence id: ').strip()

      if input_sentence_id == 'q':
        return

      print_with_time('Input Sentence: {}'.format(input_sentence_id))
      query_filter = 'GUID == "' + input_sentence_id + '"'
      input_data_object = df.query(query_filter)
      input_sentence = input_data_object['CONTENT']

      start_time = time.time()
      query_sentence_vector = generate_embeddings(input_sentence.values[0], args.use_model, random_projection_matrix)
      print_with_time('vec done')
      similar_sentences = find_similar_items(ann, query_sentence_vector, content_array, args.k)
      end_time = time.time()
      print_with_time('nns done: Time: {}'.format(end_time-start_time))
      for sentence in similar_sentences[1:]:
        if args.filter_data:
          if sentence[2] in ['country-related', 'person-related']:
            print(sentence[0])
        else:
          print(sentence[0])

if __name__ == '__main__':
    main()