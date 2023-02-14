import tensorflow as tf
import numpy as np
import cv2

class neural_style_transfer:
    def __init__(self):
        self.content_layers = ['Conv_1']

        self.style_layers = ['Conv1',
                'block_1_expand',
                'block_2_expand',
                'block_3_expand',
                'block_4_expand']               

        self.content_and_style_layers =  self.content_layers + self.style_layers

        self.NUM_CONTENT_LAYERS = len(self.content_layers)
        self.NUM_STYLE_LAYERS = len(self.style_layers)
        self.mobilenet_model()

        
    def preprocess_image(self,image):
        '''preprocesses a given image to use with Inception model'''
        image = tf.cast(image, dtype=tf.float32)
        #image = (image / 127.5) - 1.0

        return image
    
    def clip_image_values(self,image, min_value=0.0, max_value=255.0):
        '''clips the image pixel values by the given min and max'''
        return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)
    
    def mobilenet_model(self):
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights='imagenet')
        mobilenet.trainable = False
    
        output_layers = [mobilenet.get_layer(name).output for name in self.content_and_style_layers]

        self.model = tf.keras.Model(inputs=mobilenet.input, outputs=output_layers)

    
    def get_style_loss(self,features, targets):
        style_loss = tf.reduce_mean(tf.square(features - targets))
        return style_loss
    
    def get_content_loss(self,features, targets):
        content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))
        return content_loss
    
    def gram_matrix(self,input_tensor):

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 

        input_shape = tf.shape(input_tensor) 
        height = input_shape[1] 
        width = input_shape[2] 

        num_locations = tf.cast(height * width, tf.float32)

        scaled_gram = gram / num_locations
            
        return scaled_gram

    def get_style_image_features(self,image):  
        
        preprocessed_style_image = self.preprocess_image(image)

        outputs = self.model(preprocessed_style_image)

        style_outputs = outputs[:self.NUM_STYLE_LAYERS] 

        gram_style_features = [self.gram_matrix(style_layer) for style_layer in style_outputs] 
        return gram_style_features

    def get_content_image_features(self,image):
        preprocessed_content_image = self.preprocess_image(image)

        outputs = self.model(preprocessed_content_image) 

        content_outputs = outputs[self.NUM_STYLE_LAYERS:]

        return content_outputs

    def get_style_content_loss(self, style_targets, style_outputs, content_targets, 
                           content_outputs, style_weight, content_weight):

        style_loss = tf.add_n([self.get_style_loss(style_output, style_target)
                                for style_output, style_target in zip(style_outputs, style_targets)])

        content_loss = tf.add_n([self.get_content_loss(content_output, content_target)
                                for content_output, content_target in zip(content_outputs, content_targets)])

        style_loss = style_loss * style_weight / self.NUM_STYLE_LAYERS 

        content_loss = content_loss * content_weight / self.NUM_CONTENT_LAYERS 

        total_loss = style_loss + content_loss 

        return total_loss
    
    def calculate_gradients(self,image, style_targets, content_targets, 
                        style_weight, content_weight, var_weight):

        with tf.GradientTape() as tape:
            
            style_features = self.get_style_image_features(image) 
            content_features = self.get_content_image_features(image) 
            loss = self.get_style_content_loss(style_targets, style_features, content_targets, 
                                            content_features, style_weight, content_weight) 

        gradients = tape.gradient(loss, image) 

        return gradients

    def update_image_with_style(self,image, style_targets, content_targets, style_weight, 
                                var_weight, content_weight, optimizer):

        gradients = self.calculate_gradients(image, style_targets, content_targets, 
                                        style_weight, content_weight, var_weight) 

        optimizer.apply_gradients([(gradients, image)]) 

        image.assign(self.clip_image_values(image, min_value=0.0, max_value=255.0))

    def fit_style_transfer(self,style_image, content_image, style_weight=1e-2, content_weight=1e-4, 
                        var_weight=0, optimizer='adam', epochs=1, steps_per_epoch=1):

        images = []
        step = 0

        style_targets = self.get_style_image_features(style_image)

        content_targets = self.get_content_image_features(content_image)

        generated_image = tf.cast(content_image, dtype=tf.float32)
        generated_image = tf.Variable(generated_image) 

        images.append(content_image)

        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.update_image_with_style(generated_image, style_targets, content_targets, 
                                        style_weight, var_weight, content_weight, optimizer) 

                print(".", end='')

            images.append(generated_image)
            print("Train step: {}".format(step))

        generated_image = tf.cast(generated_image, dtype=tf.uint8)

        return generated_image, images