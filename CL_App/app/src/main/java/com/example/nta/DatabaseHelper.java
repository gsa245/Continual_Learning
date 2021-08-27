package com.example.nta;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import androidx.annotation.Nullable;
import com.example.nta.Database.*;

import java.sql.Blob;

public class DatabaseHelper extends SQLiteOpenHelper {

    public static final String DATABASE_NAME = "nta.db";
    public static final int DATABASE_VERSION = 1;

    public DatabaseHelper(@Nullable Context context) {
        super(context, DATABASE_NAME,null,DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        // Setting up queries for Table Creation
        final String SQL_CREATE_REPLAY_BUFFER_IMAGES_TABLE = "CREATE TABLE " +
                ReplayBufferImages.TABLE_NAME + " ("+
                ReplayBufferImages._ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                ReplayBufferImages.COLUMN_CLASS +" TEXT NOT NULL, " +
                ReplayBufferImages.COLUMN_SAMPLE_BLOB +" BLOB NOT NULL, " +
                ReplayBufferImages.COLUMN_SCENARIO + " TEXT NOT NULL " +
                ");";

        final String SQL_CREATE_TRAINING_SAMPLES_TABLE = "CREATE TABLE " +
                TrainingSamples.TABLE_NAME + " ("+
                TrainingSamples._ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                TrainingSamples.COLUMN_CLASS +" TEXT NOT NULL, " +
                TrainingSamples.COLUMN_SAMPLE_BLOB +" BLOB NOT NULL, " +
                TrainingSamples.COLUMN_SCENARIO +" TEXT NOT NULL, " +
                TrainingSamples.COLUMN_APPEARANCE + " TIMESTAMP DEFAULT CURRENT_TIMESTAMP"+
                ");";

        final String SQL_CREATE_SCENARIOS_TABLE = "CREATE TABLE " +
                Scenarios.TABLE_NAME + " ("+
                Scenarios._ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                Scenarios.COLUMN_SCENARIO +" TEXT NOT NULL, " +
                Scenarios.COLUMN_ITERATION +" INTEGER, " +
                Scenarios.COLUMN_TRAINING_SAMPLES_NUMBER +" INTEGER, " +
                Scenarios.COLUMN_REPLAY_1 +" INTEGER, " +
                Scenarios.COLUMN_REPLAY_2 +" INTEGER, " +
                Scenarios.COLUMN_REPLAY_3 +" INTEGER, " +
                Scenarios.COLUMN_REPLAY_4 +" INTEGER" +
                ");";

        final String SQL_CREATE_CLASS_BUTTON_IMAGES_TABLE = "CREATE TABLE " +
                ClassButtonImages.TABLE_NAME + " ("+
                ClassButtonImages._ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                ClassButtonImages.COLUMN_CLASS_NAME +" TEXT NOT NULL, " +
                ClassButtonImages.COLUMN_IMAGE +" BLOB NOT NULL" +
                ");";

        // Running queries for Table Creation
        db.execSQL(SQL_CREATE_REPLAY_BUFFER_IMAGES_TABLE);
        db.execSQL(SQL_CREATE_TRAINING_SAMPLES_TABLE);
        db.execSQL(SQL_CREATE_SCENARIOS_TABLE);
        db.execSQL(SQL_CREATE_CLASS_BUTTON_IMAGES_TABLE);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }

    public boolean insertReplaySample(byte[] sample, String sampleClass,String scenario)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues contentValues = new ContentValues();
        contentValues.put(ReplayBufferImages.COLUMN_CLASS, sampleClass);
        contentValues.put(ReplayBufferImages.COLUMN_SAMPLE_BLOB,sample);
        contentValues.put(ReplayBufferImages.COLUMN_SCENARIO,scenario);
        long result = db.insert(ReplayBufferImages.TABLE_NAME,null,contentValues);

        // Insert success
        return result != -1;
    }

    public boolean insertTrainingSample(byte[] sample, String sampleClass,String scenario)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues contentValues = new ContentValues();
        contentValues.put(TrainingSamples.COLUMN_CLASS, sampleClass);
        contentValues.put(TrainingSamples.COLUMN_SAMPLE_BLOB,sample);
        contentValues.put(TrainingSamples.COLUMN_SCENARIO,scenario);
        long result = db.insert(TrainingSamples.TABLE_NAME,null,contentValues);

        // Insert success
        return result != -1;
    }

    public boolean insertClassButtonImage(byte[] image, String sampleClass)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues contentValues = new ContentValues();
        contentValues.put(ClassButtonImages.COLUMN_CLASS_NAME, sampleClass);
        contentValues.put(ClassButtonImages.COLUMN_IMAGE,image);
        long result = db.insert(ClassButtonImages.TABLE_NAME,null,contentValues);

        // Insert success
        System.out.print("AEL OLE KOSTAS: "+result);
        return result != -1;
    }

    public boolean insertScenario(String scenario,int iteration,int sample_number, int replay_1,
                                  int replay_2, int replay_3, int replay_4)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues contentValues = new ContentValues();
        contentValues.put(Scenarios.COLUMN_SCENARIO, scenario);
        contentValues.put(Scenarios.COLUMN_ITERATION, iteration);
        contentValues.put(Scenarios.COLUMN_TRAINING_SAMPLES_NUMBER, sample_number);
        contentValues.put(Scenarios.COLUMN_REPLAY_1, replay_1);
        contentValues.put(Scenarios.COLUMN_REPLAY_2, replay_2);
        contentValues.put(Scenarios.COLUMN_REPLAY_3, replay_3);
        contentValues.put(Scenarios.COLUMN_REPLAY_4, replay_4);
        long result = db.insert(Scenarios.TABLE_NAME,null,contentValues);

        // Insert success
        return result != -1;
    }

    public Cursor getTrainingSamples(String scenario)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        String query = "SELECT * FROM "+TrainingSamples.TABLE_NAME+
                " WHERE "+TrainingSamples.COLUMN_SCENARIO+" = '"+scenario+"'" +
                " ORDER BY " + TrainingSamples.COLUMN_APPEARANCE + " ASC";
        Cursor res = db.rawQuery(query,null);
        return res;
    }

    public Cursor getScenario(String scenario)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        String query = "SELECT * FROM "+Scenarios.TABLE_NAME+
                " WHERE "+Scenarios.COLUMN_SCENARIO+" = '"+scenario+"'" +
                " ORDER BY " + Scenarios.COLUMN_ITERATION + " ASC";
        Cursor res = db.rawQuery(query,null);
        return res;
    }

    public Cursor getReplayBufferImages(String scenario)
    {
        SQLiteDatabase db = this.getWritableDatabase();
        String query = "SELECT * FROM "+ReplayBufferImages.TABLE_NAME+
                " WHERE "+ReplayBufferImages.COLUMN_SCENARIO+" = '"+scenario+"'";
        Cursor res = db.rawQuery(query,null);
        return res;
    }

    public Cursor getClassButtonImages()
    {
        SQLiteDatabase db = this.getWritableDatabase();
        String query = "SELECT * FROM "+ClassButtonImages.TABLE_NAME;
        Cursor res = db.rawQuery(query,null);
        return res;
    }

    public void emptyReplayBuffer(String scenario){
        SQLiteDatabase db = this.getWritableDatabase();
        int result = db.delete(ReplayBufferImages.TABLE_NAME,
                ReplayBufferImages.COLUMN_SCENARIO + "=?", new String[]{scenario});
    }

    public void emptyTrainingSamples(String scenario){
        SQLiteDatabase db = this.getWritableDatabase();
        int result = db.delete(TrainingSamples.TABLE_NAME,
                TrainingSamples.COLUMN_SCENARIO + "=?", new String[]{scenario});
    }

    public void emptyClassButtonImages(){
        SQLiteDatabase db = this.getWritableDatabase();
        int result = db.delete(ClassButtonImages.TABLE_NAME,null,null);
    }

}
