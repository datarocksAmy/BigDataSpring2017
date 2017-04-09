# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ===============================================================================================================
# CS5542 Big Data Analytics & Application Lab
# Assignment #11 - TensorFlow (Show & Tell)
# Dataset: Project Data ( KC 3 Fountains - Children's Fountain + JC Nichols + Muse of Missouri )
# 20 Chia-Hui Amy Lin
# ================================================================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

import glob

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_path", "pretrained_model/",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "pretrained_model/word_counts.txt", "Text file containing the vocabulary.")
#tf.flags.DEFINE_string("input_files", "3165123595.jpg,3164328039.jpg",
                       #"File pattern or comma-separated list of file patterns "
                       #"of image files.")

# Get all the images from the data sets
files = glob.glob("Fountain_photos/*/*.jpg")
data_sets = ",".join(str(x) for x in files)
tf.flags.DEFINE_string("input_files", data_sets,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
      filenames.extend(tf.gfile.Glob(file_pattern))
      filenames.append(file_pattern)
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    print("================================= Graph Loaded =================================")

    # Write the Captions oof each images to fountainCaptions.txt file
    with open("fountainCaptions.txt", "w") as captionsF:
        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        captionsF.write("[ Children's Fountain | J.C. Nichols Memorial Fountain | Muse of Missouri Captions ] " + "\n")
        captionsF.write("------------------------------------------------------------------------------------------------------" + "\n")
        generator = caption_generator.CaptionGenerator(model, vocab)

        for filename in filenames:
            # print(filename)

            with tf.gfile.GFile(filename, "r") as f:
                image = f.read()
                # print(image)
            captions = generator.beam_search(sess, image)
            print("Captions for image %s:" % os.path.basename(filename))
            captionsF.write("Captions for image %s:" % os.path.basename(filename) + "\n")
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                captionsF.write("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)) + "\n")
            print("\n")
            captionsF.write("\n")


if __name__ == "__main__":
  tf.app.run()
