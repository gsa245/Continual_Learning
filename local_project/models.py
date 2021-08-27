import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from sklearn.preprocessing import LabelEncoder
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter
import numpy as np
import random
import gc

class TestModel:

    def __init__(self,image_size=224):
        print("> Plain Transfer Learning Model Initiated")

        self.base = bases.MobileNetV2Base(image_size=224)

        # self.base = tf.keras.applications.MobileNetV2(
        #                 input_shape=(image_size,image_size, 3),
        #                 alpha=1.0,
        #                 weights='imagenet')

        self.head = tf.keras.Sequential([
            layers.Flatten(input_shape=(7, 7, 1280)),
            layers.Dense(
                units=32,
                activation='relu',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
            layers.Dense(
                units=4,
                activation='softmax',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
        ])

        self.head.compile(optimizer='sgd',loss='sparse_categorical_crossentropy')

    def convertToTFLite(self):
        print("> Converting model to TF Lite")

        converter = TFLiteTransferConverter(4,
                                            self.base,
                                            heads.KerasModelHead(self.head),
                                            optimizers.SGD(3e-2),
                                            train_batch_size=20)

        converter.convert_and_save('custom_model_test')

        # self.base.save("test_base", save_format='tf')
        # self.head.save("test_head", save_format='tf')

class TransferLearningModel:

    def __init__(self,image_size=224,name="None",replay_buffer=3100):
        print("> Plain Transfer Learning Model Initiated")
        # base = bases.MobileNetV2Base(image_size=224)
        self.image_size = image_size
        self.name = name
        self.replay_representations_x = []
        self.replay_representations_y = []
        self.replay_buffer = replay_buffer # The number of patterns stored

    def buildBase(self):
        self.base = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                      alpha=1.0,
                                                      include_top=False,
                                                      weights='imagenet')
        for l in self.base.layers:
            if ('_BN' in l.name):
                l.renorm = True

        self.base.trainable = False

        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        f = inputs
        f_out = self.base(f)
        self.feature_extractor = tf.keras.Model(f, f_out)
        self.feature_extractor.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='categorical_crossentropy',
                                       metrics=['accuracy'])

    def buildHead(self,sl_units=32):
        self.sl_units = sl_units
        self.head = tf.keras.Sequential([
            layers.Flatten(input_shape=(4, 4, 1280)),
            layers.Dense(
                units=sl_units,
                activation='relu',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
            layers.Dense(
                units=50,  # Number of classes
                activation='softmax',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
        ])
        # self.base.summary()

        # Test head
        # th = self.base.output
        # th_out = self.head(th)
        # self.test_head = tf.keras.Model(th,th_out)
        self.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

    def buildCompleteModel(self):
        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        x = inputs
        # we dont use inputs here
        x = self.base(x)
        outputs = self.head(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        #self.model.summary()

    # Refreshing replay memory with new samples and removing old ones if necessary
    # TODO Fix bug with patterns being a little less than they should be
    def storeRepresentations(self, train_x, train_y,random_select=False):
        replay_memory_size = len(self.replay_representations_x) + len(train_x)
        replacement_batch_size = int(self.replay_buffer * 0.015) # 1.5% sample replacement

        if random_select:
            if replay_memory_size >= self.replay_buffer:
                x_sample, y_sample = zip(*random.sample(list(zip(train_x, train_y)), replacement_batch_size))  # 150 random samples
                #print("SAS: ",np.array(train_x).shape," -> ",np.array(x_sample).shape)
                x_sample = self.feature_extractor.predict(np.array(x_sample))

                # Removing old samples
                patterns_to_delete = random.sample(range(0, len(self.replay_representations_x)-1), replacement_batch_size)
                for pat in sorted(patterns_to_delete, reverse=True):
                    del self.replay_representations_x[pat]
                    del self.replay_representations_y[pat]
            else:
                x_sample = self.feature_extractor.predict(train_x)
                y_sample = train_y
        else:
            x_sample = self.feature_extractor.predict(train_x)
            y_sample = train_y

            # Removing old samples
            if replay_memory_size >= self.replay_buffer:
                self.replay_representations_x = self.replay_representations_x[len(x_sample):len(self.replay_representations_x)]
                self.replay_representations_y = self.replay_representations_y[len(x_sample):len(self.replay_representations_y)]

        # Adding new ones
        for i in range(len(x_sample)):
            self.replay_representations_x.append(x_sample[i])
            self.replay_representations_y.append(y_sample[i])

        gc.collect()
        print("Replay X: ",len(self.replay_representations_x)," Replay Y: ",len(self.replay_representations_y))

    def replay(self):
        # TODO: Right now the model REPLAYS before or after training.
        #      We should mix the new samples with replayed ones instead
        replay_x = np.array(self.replay_representations_x)
        replay_y = np.array(self.replay_representations_y)
        # print("maria ",len(self.replay_representations_x))
        # print(replay_x.shape, "  ",replay_y.shape)

        print("> REPLAYING")
        self.head.fit(replay_x,replay_y,epochs=1,verbose=0) # just 1 epoch for now

        # for i in range(len(self.replay_representations_x)):
        #     replay_x = self.replay_representations_x[i]
        #     replay_y = self.replay_representations_y[i]
        #     self.head.fit(replay_x,replay_y,epochs=1,verbose=0) # just 1 epoch for now
