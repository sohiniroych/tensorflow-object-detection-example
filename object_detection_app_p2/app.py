#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz'

import base64
import io
import os
import pathlib
import sys
import tempfile

MODEL_BASE = '/opt/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/mscoco_label_map.pbtxt'

from decorator import requires_auth
from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from utils import label_map_util
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError

# Patch the location of gfile
tf.gfile = tf.io.gfile


app = Flask(__name__)


@app.before_request
@requires_auth
def before_request():
  pass



content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())


def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])


class ObjectDetector(object):

  def __init__(self):

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

    model_url = MODEL_URL
    base_url = os.path.dirname(model_url)+"/"
    model_file = os.path.basename(model_url)
    model_name = os.path.splitext(os.path.splitext(model_file)[0])[0]
    model_dir = tf.keras.utils.get_file(
        fname=model_name, origin=base_url + model_file, untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    self.model = model

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = self.model(input_tensor)

    output_dict = {key:value.numpy() for key,value in output_dict.items()}
    #####Debug here##########################
    boxes = tf.convert_to_tensor(output_dict['detection_boxes'][0])
    detection_masks=tf.convert_to_tensor(output_dict['detection_masks'][0])
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, boxes,image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    output_dict['detection_masks_reframed']=detection_masks_reframed.numpy()
   
    return output_dict


def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = io.BytesIO()
  image.save(image_buffer, format='PNG')
  imgstr = 'data:image/png;base64,{:s}'.format(
      base64.b64encode(image_buffer.getvalue()).decode().replace("'", ""))
  return imgstr


def detect_objects(image_path):
  image = Image.open(image_path).convert('RGB')
  image.thumbnail((840, 840), Image.ANTIALIAS)
  (im_width, im_height) = image.size
  image_np=np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  output_dict = client.detect(image)
  
  
  
  op_img=viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np.copy(),
      output_dict['detection_boxes'][0],
      (output_dict['detection_classes'][0]).astype(int),
      output_dict['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.10,
      agnostic_mode=False,
      instance_masks=output_dict.get('detection_masks_reframed',None),
      line_thickness=2)

  result = {}
  data = Image.fromarray(op_img)
  result['original']=encode_image(image.copy())
  result['objects'] = encode_image(data)
  return result


@app.route('/')
def upload():
  photo_form = PhotoForm(request.form)
  return render_template('upload.html', photo_form=photo_form, result={})


@app.route('/post', methods=['GET', 'POST'])
def post():
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))
  if request.method == 'POST' and form.validate():
    with tempfile.NamedTemporaryFile() as temp:
      form.input_photo.data.save(temp)
      temp.flush()
      result = detect_objects(temp.name)

    photo_form = PhotoForm(request.form)
    return render_template('upload.html',
                           photo_form=photo_form, result=result)
  else:
    return redirect(url_for('upload'))


client = ObjectDetector()


if __name__ == '__main__':
  if "serve" in sys.argv: app.run(host='0.0.0.0', port=8080, debug=False)
