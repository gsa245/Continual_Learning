package com.example.nta;

import android.util.Log;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * A simple class for measuring execution time in various contexts.
 * NOTE: What does execution time mean in this context? Is it the time to train using each individual
 * image?
 */
public class LoggingBenchmark {

    private static boolean ENABLED = false;

    private final String tag;

    private final Map<String,Long> totalImageTime = new HashMap<>(); // UNKNOWN USAGE
    private final Map<String, Map<String,Long>> stageTime = new HashMap<>(); // UNKNOWN USAGE
    private final Map<String,Map<String,Long>> stageStartTime = new HashMap<>(); // UNKNOWN USAGE

    LoggingBenchmark(String tag){
        this.tag = tag;
    }

    // UNKNOWN USAGE
    public void startStage(String imageId, String stageName){
        if(!ENABLED){
            return;
        }

        Map<String,Long> stageStartTimeForImage;
        if (!stageStartTime.containsKey(imageId)){
            stageStartTimeForImage = new HashMap<>();
            stageStartTime.put(imageId,stageStartTimeForImage);
        }else{
            stageStartTimeForImage = stageStartTime.get(imageId);
        }

        long timeNs = System.nanoTime();
        stageStartTimeForImage.put(stageName,timeNs);
    }

    public void endStage(String imageId, String stageName){
        if (!ENABLED){
            return;
        }

        long endTime = System.nanoTime();
        long startTime = stageStartTime.get(imageId).get(stageName);
        long duration = endTime - startTime;

        if(!stageTime.containsKey(imageId)){
            stageTime.put(imageId,new HashMap<>());
        }
        stageTime.get(imageId).put(stageName,duration);

        if (!totalImageTime.containsKey(imageId)){
            totalImageTime.put(imageId,0L);
        }
        totalImageTime.put(imageId, totalImageTime.get(imageId) + duration);
    }

    public void finish(String imageId){
        if (!ENABLED){
            return;
        }

        StringBuilder msg = new StringBuilder();

        for (Map.Entry<String, Long> entry : stageTime.get(imageId).entrySet()) {
            msg.append(String.format(Locale.getDefault(),
                    "%s: %.2fms | ", entry.getKey(), entry.getValue() / 1.0e6));
        }

        msg.append(String.format(Locale.getDefault(),
                "TOTAL: %.2fms", totalImageTime.get(imageId) / 1.0e6));
        Log.d(tag, msg.toString());
    }


}
