import tensorflow as tf
import vgg16

if __name__ == '__main__':
    vgg = vgg16.Vgg16('vgg16_pruned_0.5.npy', 'vgg16_pruned_0.5_mask.npy')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        '''
        vgg.prune(sess, 2)
        vgg.retrain(sess, 100000)
        vgg.save_weights(sess, 'vgg16_pruned_0.5.npy')
        vgg.save_mask(sess, 'vgg16_pruned_0.5_mask.npy')
        '''
        vgg.prune(sess, 2)
        vgg.retrain(sess, 100000)
        vgg.save_weights(sess, 'vgg16_pruned_1.0.npy')
        vgg.save_mask(sess, 'vgg16_pruned_1.0_mask.npy')

