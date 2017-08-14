import tensorflow as tf

original_image = tf.image.decode_png(tf.read_file('airplane.png'), channels=4)

with tf.Session() as sess:
  for x in range(0, 601, 10):
    image_cropped = tf.image.crop_to_bounding_box(original_image, 0, x, 300, 200)
    image_cropped_png = tf.image.encode_png(image_cropped)
    fname = tf.constant('images/airplane' + str(x) + '.png')
    fwrite = tf.write_file(fname, image_cropped_png)
    sess.run(fwrite)