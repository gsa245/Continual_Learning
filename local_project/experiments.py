from models import TransferLearningModel, TestModel
from data_loader import CORE50
from utils import *
import tensorflow as tf
from matplotlib import pyplot as plt
import json

class Experiments:

    def __init__(self):
        print("> Experiments Initialized")

    def plotExperiment(self,experiment_name,title):
        min_val = 100
        max_val = 50
        with open('experiments/' + experiment_name + '.json', ) as json_file:
            usecases = json.load(json_file)
            for usecase in usecases:
                for key, value in usecase.items():
                    plt.plot(value['acc'], label=key)
                    cur_min = min(value['acc'])
                    cur_max = max(value['acc'])
                    if cur_min < min_val:
                        min_val = cur_min
                    elif cur_max > max_val:
                        max_val = cur_max

        plt.title(title)
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Encountered Batches")
        plt.yticks(np.arange(round(min_val), round(max_val)+10, 5))
        plt.grid()
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(experiment_name)

    def storeExperimentOutput(self,experiment_name,usecase_name,accuracies,losses):
        data = []

        # Load previously recorded usescases
        with open('experiments/'+experiment_name+'.json', ) as json_file:
            usecases = json.load(json_file)
            for usecase in usecases:
                data.append(usecase)

        # Store new usecase
        with open('experiments/'+experiment_name+'.json', 'w') as outfile:
            exp = dict()
            exp[usecase_name] = dict()
            exp[usecase_name]["acc"] = accuracies
            exp[usecase_name]["loss"] = losses
            data.append(exp)
            json.dump(data,outfile)

    def runRandomVSFIFOReplayExperiment(self,experiment_name,usecase,replay_size=5000,random_selection=False):
        print("> Running Random VS FIFO Replay experiment")
        #dataset = CORE50(root='C:/Users/User/Desktop/CYENS_repositories/Datasets/cor50/', scenario="nicv2_391",preload=True)
        dataset = CORE50(root='/home/giorgos/cor50dataset/core50_128x128/', scenario="nicv2_391", preload=False)
        test_x, test_y = dataset.get_test_set()
        test_x = preprocess(test_x)

        # Building main model
        cl_model = TransferLearningModel(image_size=128, name=usecase,replay_buffer=replay_size)
        cl_model.buildBase()
        cl_model.buildHead(sl_units=128)
        cl_model.buildCompleteModel()

        accuracies = []
        losses = []
        temp_model = TransferLearningModel(image_size=128)

        # Training

        # loop over the training incremental batches
        for i, train_batch in enumerate(dataset):
            train_x, train_y = train_batch
            train_x = preprocess(train_x)

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            if i == 1:
                temp_model.buildHead(sl_units=cl_model.sl_units)
                temp_model.head.set_weights(cl_model.head.get_weights())
                cl_model.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                cl_model.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                cl_model.head.set_weights(temp_model.head.get_weights())

            # Reimplement these
            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], 128)
            shuffle_in_unison([train_x, train_y], in_place=True)

            # tf_model.model.fit(train_x,train_y,epochs=4)
            print("---------------------------------")

            features = cl_model.feature_extractor.predict(train_x)
            cl_model.head.fit(features, train_y, epochs=4, verbose=0)
            if i >= 1:
                cl_model.replay()
            cl_model.storeRepresentations(train_x, train_y, random_select=random_selection)

            # Evaluate temp
            loss, acc = cl_model.model.evaluate(test_x, test_y)
            accuracies.append(round(acc*100,1))
            losses.append(loss)
            print("> ", cl_model.name, " Accuracy: ", acc, " Loss: ", loss)
            print("---------------------------------")

        self.storeExperimentOutput(experiment_name=experiment_name,
                                   usecase_name=usecase,
                                   accuracies=accuracies,
                                   losses=losses)


    def runTransferLearningExperiment(self, experiment_name, usecase):
        print("> Running Random VS FIFO Replay experiment")
        # dataset = CORE50(root='C:/Users/User/Desktop/CYENS_repositories/Datasets/cor50/', scenario="nicv2_391",preload=True)
        dataset = CORE50(root='/home/giorgos/cor50dataset/core50_128x128/', scenario="nicv2_391", preload=False)
        test_x, test_y = dataset.get_test_set()
        test_x = preprocess(test_x)

        # Building main model
        tl_model = TransferLearningModel(image_size=128, name=usecase)
        tl_model.buildBase()
        tl_model.buildHead(sl_units=128)
        tl_model.buildCompleteModel()

        accuracies = []
        losses = []
        temp_model = TransferLearningModel(image_size=128)

        # Training

        # loop over the training incremental batches
        for i, train_batch in enumerate(dataset):
            train_x, train_y = train_batch
            train_x = preprocess(train_x)

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            if i == 1:
                temp_model.buildHead(sl_units=tl_model.sl_units)
                temp_model.head.set_weights(tl_model.head.get_weights())
                tl_model.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                tl_model.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                tl_model.head.set_weights(temp_model.head.get_weights())

            # Reimplement these
            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], 128)
            shuffle_in_unison([train_x, train_y], in_place=True)

            # tf_model.model.fit(train_x,train_y,epochs=4)
            print("---------------------------------")

            features = tl_model.feature_extractor.predict(train_x)
            tl_model.head.fit(features, train_y, epochs=4, verbose=0)

            # Evaluate temp
            loss, acc = tl_model.model.evaluate(test_x, test_y)
            accuracies.append(round(acc * 100, 1))
            losses.append(loss)
            print("> ", tl_model.name, " Accuracy: ", acc, " Loss: ", loss)
            print("---------------------------------")

        self.storeExperimentOutput(experiment_name=experiment_name,
                                   usecase_name=usecase,
                                   accuracies=accuracies,
                                   losses=losses)

    def runTestExperiment(self):
        print("> Running test experiment")
        #dataset = CORE50(root='C:/Users/User/Desktop/CYENS_repositories/Datasets/cor50/', scenario="nicv2_391",preload=True)
        dataset = CORE50(root='/home/giorgos/cor50dataset/core50_128x128/', scenario="nicv2_391",preload=False)
        test_x, test_y = dataset.get_test_set()
        test_x = preprocess(test_x)

        # Building main model
        model_32 = TransferLearningModel(image_size=128,name="CL_32")
        model_32.buildBase()
        model_32.buildHead(sl_units=32)
        model_32.buildCompleteModel()

        # model_64 = TransferLearningModel(image_size=128,name="CL_64")
        # model_64.buildBase()
        # model_64.buildHead(sl_units=64)
        # model_64.buildCompleteModel()

        # tf_model_64 = TransferLearningModel(image_size=128,name="TL_128")
        # tf_model_64.buildBase()
        # tf_model_64.buildHead(sl_units=128)
        # tf_model_64.buildCompleteModel()

        model_128 = TransferLearningModel(image_size=128,name="CL_128")
        model_128.buildBase()
        model_128.buildHead(sl_units=128)
        model_128.buildCompleteModel()

        model_256 = TransferLearningModel(image_size=128, name="CL_256")
        model_256.buildBase()
        model_256.buildHead(sl_units=256)
        model_256.buildCompleteModel()

        cl_models = [model_32,model_128,model_256] #removed 64
        accuracies = dict()
        losses = dict()
        accuracies[model_32.name] = []
        #accuracies[model_64.name] = []
        accuracies[model_128.name] = []
        accuracies[model_256.name] = []
        #accuracies[tf_model_64.name] = []
        losses[model_32.name] = []
        #losses[model_64.name] = []
        losses[model_128.name] = []
        losses[model_256.name] = []
        #losses[tf_model_64.name] = []

        temp_model = TransferLearningModel(image_size=128)

        # Training

        # loop over the training incremental batches
        for i, train_batch in enumerate(dataset):
            train_x, train_y = train_batch
            train_x = preprocess(train_x)

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            if i == 1:
                for cl_model in cl_models:
                    temp_model.buildHead(sl_units=cl_model.sl_units)
                    temp_model.head.set_weights(cl_model.head.get_weights())
                    cl_model.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    cl_model.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    cl_model.head.set_weights(temp_model.head.get_weights())

                # temp_model.buildHead(sl_units=128)
                # temp_model.head.set_weights(tf_model_64.head.get_weights())
                # tf_model_64.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                #                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # tf_model_64.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                #                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # tf_model_64.head.set_weights(temp_model.head.get_weights())

            # Reimplement these
            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], 128)
            shuffle_in_unison([train_x, train_y], in_place=True)

            # tf_model.model.fit(train_x,train_y,epochs=4)
            print("---------------------------------")
            for cl_model in cl_models:
                features = cl_model.feature_extractor.predict(train_x)
                cl_model.head.fit(features, train_y, epochs=4,verbose=0)
                if i >= 1:
                    cl_model.replay()
                cl_model.storeRepresentations(train_x, train_y)

                # Evaluate temp
                loss, acc = cl_model.model.evaluate(test_x, test_y)
                accuracies[cl_model.name].append(acc)
                losses[cl_model.name].append(losses)

                print("> ",cl_model.name," Accuracy: ", acc, " Loss: ",loss)

            # features = tf_model_64.feature_extractor.predict(train_x)
            # tf_model_64.head.fit(features, train_y, epochs=4,verbose=0)
            # loss, acc = tf_model_64.model.evaluate(test_x, test_y)
            # accuracies[tf_model_64.name].append(acc)
            # losses[tf_model_64.name].append(loss)
            # print("> ", tf_model_64.name, " Accuracy: ", acc, " Loss: ", loss)
            print("---------------------------------")

        for cl_model in cl_models:
            plt.plot(accuracies[cl_model.name],label=cl_model.name)

        #plt.plot(accuracies[tf_model_64.name],label=tf_model_64.name)
        plt.legend(loc=0)
        plt.show()
        plt.savefig("CL REPLAY Units Number - Buffer Limits")