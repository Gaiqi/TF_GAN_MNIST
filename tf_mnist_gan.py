import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


#Generator Network with 2 hidden layers
def generator(noise,reuse=None):
    with tf.variable_scope('gen',reuse=reuse): #allows to have subsets of parameters
        #Layer1 
        hidden1 = tf.layers.dense(inputs=noise,units=256)
        hidden1 = tf.nn.leaky_relu(hidden1,alpha = 0.01) #relu
   
        #Layer2
        hidden2 = tf.layers.dense(inputs=hidden1,units=128)
        hidden2 = tf.nn.leaky_relu(hidden2,alpha = 0.01) #relu

        #Output
        output = tf.layers.dense(inputs=hidden2,units=784,activation=tf.nn.tanh)
        return output

#Discriminator Network
def discriminator(X,reuse=None):
    with tf.variable_scope('disc',reuse=reuse): #allows to have subsets of parameters
        
        #Layer1
        hidden1 = tf.layers.dense(inputs=X,units=256)
        hidden1 = tf.nn.leaky_relu(hidden1,alpha = 0.01)  #relu   
 
        #Layer2
        hidden2 = tf.layers.dense(inputs=hidden1,units=128)
        hidden2 = tf.nn.leaky_relu(hidden2,alpha = 0.01) #relu
             
        #Output
        logits = tf.layers.dense(inputs=hidden2,units=1)
        output = tf.sigmoid(logits)
        return output, logits


#Loss Calculation
def loss_calc(logits,preds):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=preds))


def mnist_gan(mnist_dir, batch_size,epochs):
    
	mnist = input_data.read_data_sets(mnist_dir,one_hot=True)
	
	img_input = tf.placeholder(tf.float32,shape=[None,784])
	noise_ip = tf.placeholder(tf.float32,shape=[None,100])#100 random points

	Gen = generator(noise_ip)

	Disc_img_pred, Disc_img_logits = discriminator(img_input) #Training Discriminator on real images

	Disc_gen_pred, Disc_gen_logits = discriminator(Gen,reuse=True) #Training generator on fake images form Generator

	Disc_img_loss = loss_calc(Disc_img_logits,tf.ones_like(Disc_img_logits)*0.9)#Applies Smoothing for Labels
	Disc_gen_loss = loss_calc(Disc_gen_logits,tf.ones_like(Disc_img_logits))
	Total_disc_loss = Disc_img_loss + Disc_gen_loss
	Gen_loss = Disc_img_loss = loss_calc(Disc_gen_logits,tf.ones_like(Disc_gen_logits))

	lr = 0.001 #Start small


	all_vars = tf.trainable_variables()
	disc_vars = [v for v in all_vars if 'disc' in v.name]
	gen_vars = [v for v in all_vars if 'gen' in v.name]

	#Different optimizers for Disc and Gen
	Disc_optim = tf.train.AdamOptimizer(lr).minimize(Total_disc_loss, var_list = disc_vars)
	Gen_optim =  tf.train.AdamOptimizer(lr).minimize(Gen_loss, var_list = gen_vars)

	samples = []

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
	    sess.run(init)
	    for epoch in range(epochs):
	        n_batch = mnist.train.num_examples//batch_size
	        for i in range(n_batch):
	            batch = mnist.train.next_batch(batch_size)
	            disc_batch = batch[0].reshape((batch_size,784))
	            disc_batch = disc_batch*2-1 #rescale for tanh
	            
	            gen_batch = np.random.uniform(-1,1,size = (batch_size,100))
	            
	            _ =sess.run(Disc_optim, feed_dict = {img_input:disc_batch, noise_ip:gen_batch})
	            _ = sess.run(Gen_optim, feed_dict = {noise_ip:gen_batch})
	        print ("EPOCH = {}".format(epoch))
	        
	        gen_ip = np.random.uniform(-1,1,size = (1,100))
	        gen_op = sess.run(generator(noise_ip,reuse=True),feed_dict={noise_ip:gen_ip})
	        
	        samples.append(gen_op)
	        
	        #save checkpoints here
		if epoch%100==0:
                	saver.save(sess, 'model-{}.ckpt'.format(epoch))
                	print ("Saving checkpoint at epoch = {}".format(epoch))
            np.save('generated_images_array.npy',samples)       
	    #cv2.imwrite('Best_generated_sample.png',samples[-1].reshape(28,28)) #Plot image generated in final epoch


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mnist_dir", default = 'mnist_train_dir', help="Train Image Directory.")
	parser.add_argument("--batch_size",default =100,type=int, help="Default batch_size = 100.")
	parser.add_argument("--epochs",default =500,type=int, help="Default epochs = 500.")
	args = parser.parse_args()

	mnist_gan(args.mnist_dir,args.batch_size,args.epochs)
