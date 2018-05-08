# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:29:56 2018

@author: TWang

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import contextlib2
import pandas as pd
from six.moves import xrange
import tensorflow as tf

from object_detection.utils import label_map_util


from object_detection.core import standard_fields
from object_detection.utils import dataset_util


# tf.app.flags.DEFINE_string("param_name", "default_val", "description") 
tf.flags.DEFINE_string('input_annotations_csv', r'D:\TF_Try\tensorflow_models\research\oid\2017_07\validation\annotations-human-bbox.csv', 'file')
tf.flags.DEFINE_string('input_images_directory', r'D:\TF_Try\tensorflow_models\research\oid\raw_images_validation', 'location')
tf.flags.DEFINE_string('input_label_map', r'D:\TF_Try\tensorflow_models\research\oid_argumented\oid_person_label_map.pbtxt', 'file')
tf.flags.DEFINE_string('output_tf_record_path_prefix', r'D:\TF_Try\tensorflow_models\research\oid_argumented\validation_tfrecords\validation.tfrecord', 'file')
tf.flags.DEFINE_integer('num_shards', 100, 'Number of TFRecord shards')

FLAGS = tf.flags.FLAGS
    
input_annotations_csv = FLAGS.input_annotations_csv
input_images_directory = FLAGS.input_images_directory
input_label_map = FLAGS.input_label_map
output_tf_record_path_prefix = FLAGS.output_tf_record_path_prefix
num_shards = FLAGS.num_shards

print('\n input_annotations_csv :',input_annotations_csv)
print('\n input_images_directory :',input_images_directory)
print('\n input_label_map :',input_label_map)
print('\n output_tf_record_path_prefix :',output_tf_record_path_prefix)
print('\n num_shards :',num_shards)

tf.logging.set_verbosity(tf.logging.INFO)

required_flags = ['input_annotations_csv', 'input_images_directory', 'input_label_map','output_tf_record_path_prefix']

for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
        raise ValueError('Flag --{} is required'.format(flag_name))
        
label_map = label_map_util.get_label_map_dict(FLAGS.input_label_map)

all_annotations = pd.read_csv(FLAGS.input_annotations_csv)  

first = all_annotations.columns[0]

all_annotations = all_annotations.drop([first],axis=1)

all_images = tf.gfile.Glob(os.path.join(FLAGS.input_images_directory, '*.jpg')) 

all_image_ids = [os.path.splitext(os.path.basename(v))[0] for v in all_images]
                 
all_image_ids = pd.DataFrame({'ImageID': all_image_ids})

all_annotations = pd.concat([all_annotations, all_image_ids])

tf.logging.log(tf.logging.INFO, 'Found %d images...', len(all_image_ids))         
                 
with contextlib2.ExitStack() as tf_record_close_stack:
    
    '''Opens all TFRecord shards for writing and adds them to an exit stack'''
    tf_record_output_filenames = ['{}-{:05d}-of-{:05d}'.format(output_tf_record_path_prefix, idx, num_shards) for idx in range(num_shards)]

    tfrecords = [tf_record_close_stack.enter_context(tf.python_io.TFRecordWriter(file_name)) for file_name in tf_record_output_filenames]
    
    output_tfrecords = tfrecords

    for counter, image_data in enumerate(all_annotations.groupby('ImageID')):
        
        tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 1000, counter)
        
        image_id_intial, image_annotations = image_data
        
        image_path = os.path.join(FLAGS.input_images_directory, image_id_intial + '.jpg')
        
        with tf.gfile.Open(image_path,'rb') as image_file:
            encoded_image = image_file.read()
        
        '''Populates a TF Example message with image annotations from a data frame.'''
        filtered_data_frame = image_annotations[image_annotations.LabelName.isin(label_map)]
        
        #image_id = image_annotations.ImageID.iloc[0].encode('utf-8')
        
        image_id = image_annotations.ImageID.iloc[0]

        feature_map = {
                       standard_fields.TfExampleFields.object_bbox_ymin:
                           dataset_util.float_list_feature(filtered_data_frame.YMin.as_matrix()),
                       standard_fields.TfExampleFields.object_bbox_xmin:
                           dataset_util.float_list_feature(filtered_data_frame.XMin.as_matrix()),
                       standard_fields.TfExampleFields.object_bbox_ymax:
                           dataset_util.float_list_feature(filtered_data_frame.YMax.as_matrix()),
                       standard_fields.TfExampleFields.object_bbox_xmax:
                           dataset_util.float_list_feature(filtered_data_frame.XMax.as_matrix()),
		  
                       standard_fields.TfExampleFields.object_class_text:
                           dataset_util.bytes_list_feature(filtered_data_frame.LabelName.as_matrix()),

                       standard_fields.TfExampleFields.object_class_label:
                           dataset_util.int64_list_feature(filtered_data_frame.LabelName.map(lambda x: label_map[x]).as_matrix()),
			  
                       standard_fields.TfExampleFields.filename:
                           dataset_util.bytes_feature(('{}.jpg'.format(image_id)).encode('utf-8')),
                       standard_fields.TfExampleFields.source_id:
                           dataset_util.bytes_feature(image_id.encode('utf-8')),
                       standard_fields.TfExampleFields.image_encoded:
                           dataset_util.bytes_feature(encoded_image),
                      } 
                      
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_map))        
        
        if tf_example:
            shard_idx = int(image_id_intial, 16) % FLAGS.num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())      