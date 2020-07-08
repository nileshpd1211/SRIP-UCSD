# import tensorflow as tf
# model=tf.keras.models.load_model("./newmodel.h5",compile=True)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model) 


import tensorflow as tf
from tensorflow.keras.models import load_model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
 

loaded_model=tf.keras.models.load_model("./newmodel.h5",compile=True)

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = [tf.int8]
converter.inference_output_type = [tf.int8]
# converter.target_spec.supported_types = [tf.compat.v1.lite.constants.INT8]
# converter.inference_type = tf.compat.v1.lite.constants.INT8
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
open("quant_newmodel.tflite", "wb").write(tflite_quant_model)
