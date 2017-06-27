import numpy as np
import tensorflow as tf

import vgg16
import utils

img_dir = 'ILSVRC2012_img_val'
img_label = 'ILSVRC2012_validation_ground_truth.txt'
img_prefix = 'ILSVRC2012_val_'
img_suffix = '.JPEG'
img_total_num = 50000
filename_synset_for_model = 'synset.txt'
filename_synset_for_label = 'synset_original.txt'
synset_for_model = None
synset_for_label = None

def val(sess, i, vgg, img_placeholder, labels):
	img = utils.load_image('{}/{}{:08d}{}'.format(img_dir, img_prefix, i + 1, img_suffix))
	img = img.reshape((1, 224, 224, 3))
	feed_dict = {img_placeholder: img}

	prob = sess.run(vgg.prob, feed_dict=feed_dict)
	top5 = np.argsort(prob[0])[-5:][::-1]
	top5 = [synset_for_model[idx] for idx in top5.tolist()]

	label = labels[i]
	label = synset_for_label[label - 1]

	in_top1 = (label == top5[0])
	in_top5 = (label in top5)

	return in_top1, in_top5

if __name__ == '__main__':
	with tf.Session() as sess:
		
		images = tf.placeholder("float", [1, 224, 224, 3])
		vgg = vgg16.Vgg16()
		with tf.name_scope("content_vgg"):
			vgg.build(images)

		label_file = open(img_label, 'r')
		labels = [int(row) for row in label_file]

		file = open(filename_synset_for_model, 'r')
		synset_for_model = [row.split(' ')[0] for row in file]
		file = open(filename_synset_for_label, 'r')
		synset_for_label = [row.split('\n')[0] for row in file]

		num_in_top1 = 0
		num_in_top5 = 0

		for i in range(img_total_num):
			in_top1, in_top5 = val(sess, i, vgg, images, labels)
			num_in_top1 += in_top1
			num_in_top5 += in_top5
			print '{}%, Top1 Accuracy: {}, Top5 Accuracy: {}'.format(
				100 * (i + 1) / img_total_num,
				float(num_in_top1) / (i + 1), 
				float(num_in_top5) / (i + 1))

