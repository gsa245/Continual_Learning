package com.example.nta;

import android.content.Context;
import android.database.Cursor;
import android.os.ConditionVariable;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.tensorflow.lite.examples.transfer.api.AssetModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.GatheringByteChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

// For now just a Trasnfer learning model
public class ContinualLearningModel implements Closeable {
    public static final int IMAGE_SIZE = 224;

    public final TransferLearningModel model;

    private final ConditionVariable shouldTrain = new ConditionVariable();
    private volatile TransferLearningModel.LossConsumer lossConsumer;

    private Button btnStoreSamples;
    private int samplesAdded = 0; // Number of samples already added to the replay buffer
    //private int trainingSamplesStored = 0; // Number of training samples already stored in DB
    private DatabaseHelper databaseHelper;
    private Context context;
    public HashMap<String,ArrayList<byte[]>> replayBuffer = new HashMap<>();


    ContinualLearningModel(Context context) {
        databaseHelper = new DatabaseHelper(context);
        this.context = context;
        model =
                new TransferLearningModel(
                        new AssetModelLoader(context, "mobilenet_softmax_model"), Arrays.asList("1", "2", "3", "4"));

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
    public Future<Void> addSample(float[] image, String className) {
        return model.addSample(image, className);
    }

    // Adds new Training sample that was stored in the local DB
    public Future<Void> addSample(ByteBuffer bottleneck, String className) {
        return model.addSample(bottleneck, className);
    }

    // This method is thread-safe, but blocking.
    public TransferLearningModel.Prediction[] predict(float[] image) {
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
    public void enableTraining(TransferLearningModel.LossConsumer lossConsumer) {
        this.lossConsumer = lossConsumer;
        shouldTrain.open();
        replay(null);
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


    // NEW FUNCTIONS FOR REPLAY


    /***
     * Adds new samples to the replay buffer. Everything
     */
    public void updateReplayBuffer(String scenario) {
        int startingPoint = samplesAdded;
        for (int i = startingPoint; i < model.trainingSamples.size(); i++) {
            ByteBuffer bottleneck = model.trainingSamples.get(i).bottleneck;
            String className = model.trainingSamples.get(i).className;
            byte[] b = new byte[bottleneck.remaining()];
            bottleneck.get(b);
            boolean success = databaseHelper.insertReplaySample(b, className,(scenario == null) ? "default" : scenario);
            System.out.println("AEL DB INSERT: " + success);
            samplesAdded++;
        }
        Toast.makeText(context, "REPLAY BUFFER SIZE: " + (model.trainingSamples.size() - startingPoint), Toast.LENGTH_SHORT).show();
    }

    /***
     * Adds new samples to buffer - normal distribution between classes - fixed number
     */
    public void updateReplayBufferSmart(String scenario){
        Toast.makeText(context, "UPDATING REPLAY BUFFER - PLEASE WAIT", Toast.LENGTH_SHORT).show();
        databaseHelper.emptyReplayBuffer((scenario == null) ? "default" : scenario);
        replayBuffer.clear();

        Cursor res = databaseHelper.getTrainingSamples((scenario == null) ? "default" : scenario);
        if (res.getCount() != 0) {
            while (res.moveToNext())
            {
                String className = res.getString(1);
                byte[] blobBytes = res.getBlob(2);
                if (!replayBuffer.containsKey(className)){
                    replayBuffer.put(className,new ArrayList<>());
                }
                replayBuffer.get(className).add(blobBytes);
            }
        } else {
            System.out.println("AEL: DEN DOYLEUEI RESTORED");
        }

//        for (int i = 0; i < model.trainingSamples.size(); i++) {
//            ByteBuffer bottleneck = model.trainingSamples.get(i).bottleneck;
//            String className = model.trainingSamples.get(i).className;
//            byte[] b = new byte[bottleneck.remaining()];
//            bottleneck.get(b);
//            if (!sampleMap.containsKey(className)){
//                sampleMap.put(className,new ArrayList<>());
//            }
//            sampleMap.get(className).add(b);
//        }

        // Inserting to database
        int replaySamplesAdded = 0;
        for (Map.Entry<String, ArrayList<byte[]>> entry : replayBuffer.entrySet()) {
            String className = entry.getKey();
            ArrayList<byte[]> classSamples = entry.getValue();
            Collections.shuffle(classSamples); // Adds randomness to the replay sample selection
            for (byte[] sample : classSamples) {
                boolean success = databaseHelper.insertReplaySample(sample, className,(scenario == null) ? "default" : scenario);
                replaySamplesAdded++;
                if (replaySamplesAdded % 10 == 0)
                    break;
            }
        }
        Toast.makeText(context, "REPLAY BUFFER SIZE: " + replaySamplesAdded, Toast.LENGTH_SHORT).show();
    }

    public void resetModelWeights(GatheringByteChannel outputChannel){
        ByteBuffer[] test = new ByteBuffer[0];
        try {
            outputChannel.write(test);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Stores a training sample into the local database - Extra function - Temporary
    // For now this is almost identical to the updateReplayBuffer() method
//    public void storeTrainingSample(String scenario) {
//        if (model.trainingSamples.size() == 0){
//            trainingSamplesStored = 0;
//        }
//        int startingPoint = trainingSamplesStored;
//        for (int i = startingPoint; i < model.trainingSamples.size(); i++) {
//            ByteBuffer bottleneck = model.trainingSamples.get(i).bottleneck;
//            String className = model.trainingSamples.get(i).className;
//            byte[] b = new byte[bottleneck.remaining()];
//            bottleneck.get(b);
//            boolean success = databaseHelper.insertTrainingSample(b, className,(scenario == null) ? "default" : scenario);
//            System.out.println("AEL DB INSERT TRAINING SAMPLE: " + success);
//            trainingSamplesStored++;
//        }
//        //Toast.makeText(context, startingPoint+ " - Samples added: " + (model.trainingSamples.size() - startingPoint), Toast.LENGTH_SHORT).show();
//    }

    /***
     * Replays samples stored in the buffer (before training)
     * TODO Replay mixed with new samples instead of before training
     */
    public void replay(String scenario) {
        Cursor res = databaseHelper.getReplayBufferImages((scenario == null) ? "default" : scenario);
        Toast.makeText(context, "REPLAYING: " + res.getCount()+" SAMPLES", Toast.LENGTH_SHORT).show();
        if (res.getCount() != 0) {
            while (res.moveToNext()) {
                String className = res.getString(1);
                byte[] blobBytes = res.getBlob(2);
                ByteBuffer bottleneck = ByteBuffer.wrap(blobBytes);
                addSample(bottleneck, className);
            }
        } else {
            System.out.println("AEL: DEN DOYLEUEI");
        }
    }
}
