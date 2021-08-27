/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.nta;

import android.content.Context;
import android.database.Cursor;
import android.os.ConditionVariable;
import android.widget.Toast;

import org.tensorflow.lite.examples.transfer.api.AssetModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.GatheringByteChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * App-layer wrapper for {@link TransferLearningModel}.
 *
 * <p>This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of {@link TransferLearningModel}.
 */
public class TransferLearningModelWrapper implements Closeable {
    public static final int IMAGE_SIZE = 224;

    public final TransferLearningModel model;

    private final ConditionVariable shouldTrain = new ConditionVariable();
    private volatile LossConsumer lossConsumer;

    private DatabaseHelper databaseHelper;
    private Context context;

    private int trainingSamplesStored = 0; // Number of training samples already stored in DB


    TransferLearningModelWrapper(Context context) {
        model =
                new TransferLearningModel(
                        new AssetModelLoader(context, "model"), Arrays.asList("1", "2", "3", "4"));

        databaseHelper = new DatabaseHelper(context);
        this.context = context;

        new Thread(() -> {
            while (!Thread.interrupted()) {
                shouldTrain.block();
                try {
                    model.train(1, lossConsumer).get();
                } catch (ExecutionException e) {
                    throw new RuntimeException("Exception occurred during model training", e.getCause());
                } catch (InterruptedException e) {
                    // no-op
                }
            }
        }).start();
    }

    // This method is thread-safe.
    // TODO Add sample is the key to using the REPLAY buffer for continual learning
    public Future<Void> addSample(float[] image, String className) {
        return model.addSample(image, className);
    }

    // Adds new Training sample that was stored in the local DB
    public Future<Void> addSample(ByteBuffer bottleneck, String className) {
        return model.addSample(bottleneck, className);
    }


    // This method is thread-safe, but blocking.
    public Prediction[] predict(float[] image) {
        return model.predict(image);
    }

    public int getTrainBatchSize() {
        return model.getTrainBatchSize();
    }

    /**
     * Start training the model continuously until {@link #disableTraining() disableTraining} is
     * called.
     *
     * @param lossConsumer callback that the loss values will be passed to.
     */
    public void enableTraining(LossConsumer lossConsumer) {
        this.lossConsumer = lossConsumer;
        shouldTrain.open();
    }

    /**
     * Stops training the model.
     */
    public void disableTraining() {
        shouldTrain.close();
    }

    /**
     * Frees all model resources and shuts down all background threads.
     */
    public void close() {
        model.close();
    }

    // Retrieves and retrains the model based on previously stored samples
    // This was a temporary solution, not currently used
      public void restoreModel(CameraFragmentViewModel cameraViewModel, String scenario){
        Cursor res = databaseHelper.getTrainingSamples((scenario == null) ? "default" : scenario);
        Toast.makeText(context, "AEL RESTORED SAMPLES: " + res.getCount(), Toast.LENGTH_SHORT).show();
        if (res.getCount() != 0) {
            while (res.moveToNext())
            {
                String className = res.getString(1);
                byte[] blobBytes = res.getBlob(2);
                ByteBuffer bottleneck = ByteBuffer.wrap(blobBytes);
                addSample(bottleneck, className);
            }
        } else {
            System.out.println("AEL: DEN DOYLEUEI RESTORED");
        }
    }

    public void storeTrainingSample(String scenario) {
        if (model.trainingSamples.size() == 1){
            trainingSamplesStored = 0;
        }
        int startingPoint = trainingSamplesStored;
        for (int i = startingPoint; i < model.trainingSamples.size(); i++) {
            ByteBuffer bottleneck = model.trainingSamples.get(i).bottleneck;
            String className = model.trainingSamples.get(i).className;
            byte[] b = new byte[bottleneck.remaining()];
            bottleneck.get(b);
            boolean success = databaseHelper.insertTrainingSample(b, className,(scenario == null) ? "default" : scenario);
            trainingSamplesStored++;
        }
        //Toast.makeText(context, startingPoint+ " - Samples added: " + (model.trainingSamples.size() - startingPoint), Toast.LENGTH_SHORT).show();
    }

    public void resetModelWeights(GatheringByteChannel outputChannel){
        ByteBuffer[] test = new ByteBuffer[0];
        try {
            outputChannel.write(test);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
