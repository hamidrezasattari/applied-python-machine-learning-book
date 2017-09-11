import tensorflow as tf, sys

image_test_path = sys.argv[1]

#Read the input image
image_test_data = tf.gfile.FastGFile(image_test_path,'rb').read()

# Load label file

label_retrained_lines = [line.rstrip() for line
     in tf.gfile.GFile("/test_files/test_retrain/updated_retrained_labels.txt")]

#Load model

with tf.gfile.FastGFile("/test_files/test_retrain/updated_retrained_graph.pb", 'rb') as f:
 graph_retrained_def = tf.GraphDef()
 graph_retrained_def.ParseFromString(f.read())
 g_retrained_in = tf.import_graph_def(graph_retrained_def, name='')

with tf.Session() as sess:
 softmax_final_layer = sess.graph.get_tensor_by_name('final_result:0')
 predictions = sess.run(softmax_final_layer, \
  {'DecodeJpeg/contents:0' : image_test_data})
 top_n_probilities = predictions[0].argsort()[-len(predictions[0]):][::-1]

for node_id in top_n_probilities:
 readable_string = label_retrained_lines[node_id]
 score = predictions[0][node_id]
 print ('%s (score %.5f)' % (readable_string,score))

