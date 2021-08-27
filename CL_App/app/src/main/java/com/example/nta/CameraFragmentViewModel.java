package com.example.nta;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.widget.ImageView;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MediatorLiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.Transformations;
import androidx.lifecycle.ViewModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class CameraFragmentViewModel extends ViewModel {

    /**
     * Current state of Training process
     */
    public enum TrainingState {
        NOT_STARTED,
        STARTED,
        PAUSED
    }

    // Whether the model training is not yet started, already started or temporarily paused.
    private MutableLiveData<TrainingState> trainingState =
            new MutableLiveData<>(TrainingState.NOT_STARTED);

    private MutableLiveData<Map<String,Integer>> numSamples
            = new MutableLiveData<>(new TreeMap<>()); // Number of ADDED samples for each class
    private MutableLiveData<Map<String,Integer>> numVisualSamples
            = new MutableLiveData<>(new TreeMap<>()); // Number of TOTAL added samples (only visualizations)

    private MutableLiveData<Integer> trainBatchSize = new MutableLiveData<>(0); // Number of samples in a single training batch
    private MutableLiveData<Boolean> trainingNewIteration = new MutableLiveData<>(false); // How many times we pressed train in a single session
    private MutableLiveData<Integer> prevIterationTotalSamples = new MutableLiveData<>(0);

    private MutableLiveData<Boolean> captureMode = new MutableLiveData<>(false); // Is capture mode enabled?
    private MutableLiveData<Boolean> continualLearningMode = new MutableLiveData<>(false); // Is CL enabled?

    private MutableLiveData<Float> lastLoss = new MutableLiveData<>(); // Last loss reported by the training routine Transfer Learning
    private MutableLiveData<Float> lastLossCL = new MutableLiveData<>(); // Last loss reported by the training routing Continual Learning
    private MutableLiveData<Boolean> inferenceSnackbarWasDisplayed = new MutableLiveData<>(false);

    // Confidence values for each class during inference
    private MutableLiveData<Map<String,Float>> confidence = new MutableLiveData<>(new TreeMap<>());

    private LiveData<Integer> totalSamples; // Total number of samples added for all classes
    private LiveData<Integer> neededSamples;
    private LiveData<String> firstChoice; // Name of the class with the highest confidence score
    private LiveData<String> secondChoice; // Name of the class with the second highest conf. score

    private HashMap<String,Bitmap> classBtnImages = new HashMap<>();

    public void resetView(){
        numVisualSamples.getValue().clear();
        numSamples.getValue().clear();
        confidence.getValue().clear();
        trainingNewIteration.postValue(false);
        prevIterationTotalSamples.postValue(0);
    }

    public HashMap<String,Bitmap> getClassBtnImages() {return classBtnImages;}

    public void updateClassBtnImage(ImageView imageView,Bitmap bitmap){
        imageView.setImageBitmap(bitmap);
    }

    public MutableLiveData<Boolean> getTrainingIteration() { return trainingNewIteration;}
    public void setTrainingIteration(boolean newValue) { trainingNewIteration.postValue(newValue);}
    public MutableLiveData<Integer> getPrevIterationTotalSamples() { return prevIterationTotalSamples;}
    public void setPrevIterationTotalSamples(int newValue) { prevIterationTotalSamples.postValue(newValue);}

    // Batch size functions
    public MutableLiveData<Integer> getTrainBatchSize() { return trainBatchSize; }
    public void setTrainBatchSize(int newValue) { trainBatchSize.postValue(newValue);}

    // Training state functions
    public LiveData<TrainingState> getTrainingState() { return  trainingState; }
    public void setTrainingState(TrainingState newValue) { trainingState.postValue(newValue);}

    // Capture Mode functions
    public MutableLiveData<Boolean> getCaptureMode() { return captureMode; }
    public void setCaptureMode(boolean newValue) { captureMode.postValue(newValue); }

    public MutableLiveData<Boolean> getContinualLearningMode() { return continualLearningMode; }
    public void setContinualLearningMode(boolean newValue) { continualLearningMode.postValue(newValue); }

    // Loss functions
    public LiveData<Float> getLastLoss() { return lastLoss; }
    public void setLastLoss(float newLoss) { lastLoss.postValue(newLoss); }
    public LiveData<Float> getLastLossCL() { return lastLossCL; }
    public void setLastLossCL(float newLoss) { lastLossCL.postValue(newLoss); }

    public LiveData<Map<String,Float>> getConfidence() { return confidence; }

    public void setConfidence(String className, float confidenceScore){
        Map<String,Float> map = confidence.getValue();
        map.put(className, confidenceScore);
        confidence.postValue(map);
    }

    public LiveData<String> getFirstChoice(){
        if (firstChoice == null){
            firstChoice = Transformations.map(confidence, map -> {
                if (map.isEmpty()){
                    return null;
                }
                return mapEntriesDecreasingValue(map).get(0).getKey();
            });
        }
        return firstChoice;
    }

    public LiveData<String> getSecondChoice(){
        if (secondChoice == null){
            secondChoice = Transformations.map(confidence, map -> {
               if (map.size() < 2){
                   return null;
               }
               return mapEntriesDecreasingValue(map).get(1).getKey();
            });
        }
        return secondChoice;
    }

    public LiveData<Map<String,Integer>> getNumSamples() {return  numSamples;}
    public LiveData<Map<String,Integer>> getNumVisualSamples() {return numVisualSamples;}

    // Returns total num of samples for all classes - NOT SURE ABOUT THAT
    public LiveData<Integer> getTotalSamples(){
         if (totalSamples == null){
             totalSamples = Transformations.map(getNumSamples(), map -> {
                int total = 0;
                for (int number : map.values()){
                    total += number;
                }
                return total;
             });
         }
         return totalSamples;
    }

    /***
     * The number of samples needed to complete a single batch (20 in begining)
     * @return
     */
    public LiveData<Integer> getNeededSamples(){
        if (neededSamples == null){
            MediatorLiveData<Integer> result = new MediatorLiveData<>();
            result.addSource(
                    getTotalSamples(),
                    totalSamples -> {
                        result.setValue(Math.max(0, getTrainBatchSize().getValue() - totalSamples + getPrevIterationTotalSamples().getValue())); // This might go negative if you give more samples than you need I think
                    });
            result.addSource(
                    getTrainBatchSize(),
                    trainBatchSize -> {
                       result.setValue(Math.max(0, trainBatchSize - getTotalSamples().getValue()) + getPrevIterationTotalSamples().getValue());
                    });
            result.addSource(
                    getTrainingIteration(),
                    trainingIteration -> {
                        result.setValue(Math.max(0, getTrainBatchSize().getValue())); // This might go negative if you give more samples than you need I think
                    });

            neededSamples = result;
        }
        return neededSamples;
    }

    public void increaseNumSamples(String className) {
        Map<String, Integer> map = numSamples.getValue();
        int currentNumber;
        if (map.containsKey(className)) {
            currentNumber = map.get(className);
        } else {
            currentNumber = 0;
        }
        map.put(className, currentNumber + 1);
        numSamples.postValue(map);
        increaseNumVisualSamples(className);
    }

    public void increaseNumVisualSamples(String className) {
        Map<String, Integer> map = numVisualSamples.getValue();
        int currentNumber;
        if (map.containsKey(className)) {
            currentNumber = map.get(className);
        } else {
            currentNumber = 0;
        }
        map.put(className, currentNumber + 1);
        numVisualSamples.postValue(map);
    }

    /**
     * Whether "you can switch to inference mode now" snackbar has been shown before.
     */
    public MutableLiveData<Boolean> getInferenceSnackbarWasDisplayed() {
        return inferenceSnackbarWasDisplayed;
    }

    public void markInferenceSnackbarWasCalled() {
        inferenceSnackbarWasDisplayed.postValue(true);
    }

    // EXTRA UTILS

    /**
     * Sorts the Map (confidence) in decending order and returns it as a list
     * @param map
     * @return
     */
    private static List<Map.Entry<String,Float>> mapEntriesDecreasingValue(Map<String,Float> map){
        List<Map.Entry<String,Float>> entryList = new ArrayList<>(map.entrySet());
        Collections.sort(entryList,(e1,e2) -> -Float.compare(e1.getValue(),e2.getValue()));
        return entryList;
    }


}