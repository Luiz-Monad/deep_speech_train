
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import multiprocessing as mp

from sox.file_info import duration as soxi_duration
from sox.core import SoxiError

def init():
  return

def duration(file):
  for i in range(1, 3):
    try:
      return soxi_duration(file)
    except SoxiError as sex:
      return sys.maxsize
    except Exception as ex:
      time.sleep(0.015)
  return sys.maxsize

if __name__ == "__main__":
  mp.set_start_method('spawn')

  from absl import app as absl_app
  from absl import flags as absl_flags
  from absl import testing as abs_testing
  from absl import logging

  import tensorflow as tf
  import pandas as pd 

  from data import download as data_download
  import deep_speech

  logging.set_verbosity(logging.INFO)
  
  data_dir = "x:/tmp/librispeech_data"
  model_dir = "x:/tmp/deep_speech_model/"
  export_dir = "x:/tmp/deep_speech_saved_model/"

  # Script to run deep speech model to achieve the MLPerf target (WER = 0.23)
  # Step 1: download the LibriSpeech dataset.
  
  logging.info("Data downloading...")
  data_download.define_data_download_flags()
  data_download.absl_flags.FLAGS.data_dir = data_dir
  data_download.absl_flags.FLAGS.mark_as_parsed()
  data_download.main(0)

  ## After data downloading, the dataset directories are:
  train_clean_100 = os.path.join(data_dir, "train-clean-100/LibriSpeech/train-clean-100.csv")
  train_clean_360 = os.path.join(data_dir, "train-clean-360/LibriSpeech/train-clean-360.csv")
  train_other_500 = os.path.join(data_dir, "train-other-500/LibriSpeech/train-other-500.csv")
  dev_clean = os.path.join(data_dir, "dev-clean/LibriSpeech/dev-clean.csv")
  dev_other = os.path.join(data_dir, "dev-other/LibriSpeech/dev-other.csv")
  test_clean = os.path.join(data_dir, "test-clean/LibriSpeech/test-clean.csv")
  test_other = os.path.join(data_dir, "test-other/LibriSpeech/test-other.csv")

  # Step 2: generate train dataset and evaluation dataset
  train_file = os.path.join(data_dir, "train_dataset.csv")
  eval_file = os.path.join(data_dir, "eval_dataset.csv")

  def concat_csv(files, dest):
    if tf.io.gfile.exists(dest):
      logging.info(
          "Already processed %s" % dest)
      return
    dfs = [pd.read_csv(filename, sep="\t") for filename in files]
    df = pd.concat(dfs)
    df.to_csv(dest, index=False, sep="\t")

  logging.info("Data preprocessing...")
  concat_csv([ train_clean_100, train_clean_360, train_other_500 ], train_file)
  concat_csv([ dev_clean, dev_other ], eval_file)

  # Step 3: filter out the audio files that exceed max time duration.
  final_train_file = os.path.join(data_dir, "final_train_dataset.csv")
  final_eval_file = os.path.join(data_dir, "final_eval_dataset.csv")

  def filter_csv(src, dest, proj, filt):
    if tf.io.gfile.exists(dest):
      logging.info(
          "Already filtered %s" % dest)
      return
    pool = mp.Pool(mp.cpu_count() * 4, init)
    df = pd.read_csv(src, sep="\t")
    files = list(map(lambda row: proj(row), df.itertuples(index=False)))
    res = pool.map_async(duration, files)
    data = dict(zip(files, res.get()))
    data = list(filter(lambda row: filt(data[proj(row)]), df.itertuples(index=False)))
    df = pd.DataFrame(data=data, columns=df.columns)
    df.to_csv(dest, index=False, sep="\t")
    pool.close()

  logging.info("Filter max time duration...")
  MAX_AUDIO_LEN = 27.0
  fileName = lambda rec: rec.wav_filename
  maxLen = lambda s: s <= MAX_AUDIO_LEN
  filter_csv(train_file, final_train_file, fileName, maxLen)
  filter_csv(eval_file, final_eval_file, fileName, maxLen)

  # Step 4: run the training and evaluation loop in background, and save the running info to a log file
  logging.info("Model training and evaluation...")
  start = time.perf_counter()

  deep_speech.define_deep_speech_flags()
  deep_speech.absl_flags.FLAGS.train_data_dir = final_train_file
  deep_speech.absl_flags.FLAGS.eval_data_dir = final_eval_file
  deep_speech.absl_flags.FLAGS.num_gpus = -1
  deep_speech.absl_flags.FLAGS.wer_threshold = 0.23
  deep_speech.absl_flags.FLAGS.seed = 1
  deep_speech.absl_flags.FLAGS.model_dir = model_dir
  deep_speech.absl_flags.FLAGS.export_dir = export_dir
  deep_speech.absl_flags.FLAGS.mark_as_parsed()
  deep_speech.main(0)

  end = time.perf_counter()
  runtime = round((end - start), 0)
  logging.info(("Model training time is %d seconds." % runtime))
