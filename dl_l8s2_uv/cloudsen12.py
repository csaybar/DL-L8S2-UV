import pkg_resources
import tensorflow as tf
from dl_l8s2_uv import model

weights_model = pkg_resources.resource_filename('dl_l8s2_uv', 'weights/landsatbiomeRGBISWIR7.hdf5')
model_clouds = model.load_model((None, None), weight_decay=0, bands_input=len([1, 2, 3, 8]))
model_clouds.load_weights(weights_model)

cloudsen12_shape = (511, 511, 4)
demo = tf.random.uniform(shape=[1, 256, 256, 4])
model_clouds(demo).shape
