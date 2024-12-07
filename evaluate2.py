from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import tensorflow as tf
BATCH_SIZE = 1
DEVICE = '/cpu:0'

class KModel(object):

    def __init__(self, model_filepath):
        self.img_shape = (480, 640,3)
        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)
        
    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.compat.v1.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        
        with self.graph.as_default():
            # Define input tensor
            self.batch_shape = (BATCH_SIZE,) + self.img_shape
            self.input  = tf.compat.v1.placeholder(tf.float32, shape=self.batch_shape,
                                            name='img_placeholder')
            # X = np.zeros(batch_shape, dtype=np.float32)
            tf.compat.v1.import_graph_def(graph_def, {'img_placeholder':self.input })

        self.graph.finalize()

        print('Model loading complete!')

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably. 
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.compat.v1.Session(graph = self.graph)

    def K_prep(self, in_path, out_path):
        files = list_files(in_path)
        full_in = [os.path.join(in_path,x) for x in files]
        full_out = [os.path.join(out_path,x) for x in files]
        # device = DEVICE
        batch_size = BATCH_SIZE
        return full_in, full_out, batch_size

    def test(self, in_path, out_path):
        data_in, paths_out, batch_size = self.K_prep(in_path, out_path)

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/add_37:0")
        # output = self.sess.run(output_tensor, feed_dict = {self.input: data})

        is_paths = type(data_in[0]) == str
        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(self.batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == self.img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            # _preds = self.sess.run(preds, feed_dict={img_placeholder:X})
            _preds = self.sess.run(output_tensor, feed_dict={self.input:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]

        if len(remaining_in) > 0:
            self.test(remaining_in, remaining_out, batch_size=1)
        # return output


if __name__ == '__main__':
    pass
