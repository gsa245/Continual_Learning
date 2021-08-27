package com.example.nta;

import android.provider.BaseColumns;

public class Database {

    public static final class ReplayBufferImages implements BaseColumns
    {
        public static final String TABLE_NAME = "replay_buffer_images";
        public static final String COLUMN_CLASS ="class";
        public static final String COLUMN_SAMPLE_BLOB = "sample";
        public static final String COLUMN_SCENARIO = "scenario_name";
    }

    public static final class TrainingSamples implements BaseColumns
    {
        public static final String TABLE_NAME = "training_samples";
        public static final String COLUMN_CLASS = "class";
        public static final String COLUMN_SAMPLE_BLOB = "sample";
        public static final String COLUMN_SCENARIO = "scenario_name";
        public static final String COLUMN_APPEARANCE = "appearance_timestamp";
    }

    public static final class Scenarios implements BaseColumns
    {
        public static final String TABLE_NAME = "scenarios";
        public static final String COLUMN_SCENARIO = "scenario_name";
        public static final String COLUMN_ITERATION = "iteration";
        public static final String COLUMN_TRAINING_SAMPLES_NUMBER = "training_samples_number";
        public static final String COLUMN_REPLAY_1 = "replay_class_1";
        public static final String COLUMN_REPLAY_2 = "replay_class_2";
        public static final String COLUMN_REPLAY_3 = "replay_class_3";
        public static final String COLUMN_REPLAY_4 = "replay_class_4";
    }

    public static final class ClassButtonImages implements BaseColumns
    {
        public static final String TABLE_NAME = "class_btn_images";
        public static final String COLUMN_CLASS_NAME = "class_name";
        public static final String COLUMN_IMAGE = "image";
    }

}
