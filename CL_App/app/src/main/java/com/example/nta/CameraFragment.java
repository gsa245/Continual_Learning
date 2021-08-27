package com.example.nta;

import androidx.camera.core.CameraX;
import androidx.camera.core.CameraX.LensFacing;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.lifecycle.ViewModelProvider;

import android.content.Context;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.os.Handler;
import android.os.HandlerThread;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.databinding.BindingAdapter;
import androidx.databinding.DataBindingUtil;
//import org.tensorflow.lite.examples.transfer.databinding.CameraFragmentBinding;
import com.example.nta.databinding.CameraFragmentBinding;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;

import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.snackbar.Snackbar;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;

public class CameraFragment extends Fragment {

    private static final int LOWER_BYTE_MASK = 0xFF; // UNKNOWN USAGE
    private static final String TAG = CameraFragment.class.getSimpleName(); // UNKNOWN USAGE
    private static final LensFacing LENS_FACING = LensFacing.BACK; // Where the Lens should be facing

    private TextureView viewFinder; // UNKNOWN USAGE
    private Integer viewFinderRotation = null; // The current rotation of the screen
    private Size bufferDimens = new Size(0, 0); // UNKNOWN USAGE
    private Size viewFinderDimens = new Size(0, 0); // UNKNOWN USAGE

    private CameraFragmentViewModel mViewModel; // The view model of this fragment
    private TransferLearningModelWrapper tlModel; // Our Transfer Lerning TF Lite model
    private ContinualLearningModel clModel; // Our Continual Learning TF lite model

    // When the user presses the "add sample" button for some class,
    // that class will be added to this queue. It is later extracted by
    // InferenceThread and processed.
    private final ConcurrentLinkedQueue<String> addSampleRequests = new ConcurrentLinkedQueue<>();

    // Measuring execution time for each image during inference?
    private final LoggingBenchmark inferenceBenchmark = new LoggingBenchmark("InferenceBench");

    private int trainingIteration = 0; // Used for Recording the scenario and storing it to the DB
    private DatabaseHelper databaseHelper;

    @Override
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        tlModel = new TransferLearningModelWrapper(getActivity());
        clModel = new ContinualLearningModel(getActivity());
        mViewModel = new ViewModelProvider(this).get(CameraFragmentViewModel.class);
        mViewModel.setTrainBatchSize(tlModel.getTrainBatchSize());
        databaseHelper = new DatabaseHelper(getActivity());

        // Retraining Model from stored database samples - TEMPORARILY DISABLED (REPLACED BY SCENARIOS)
        //tlModel.restoreModel(mViewModel,null);
        //clModel.restoreModel(mViewModel,null);

        try {
            tlModel.model.loadParameters(getActivity().openFileInput("tlmodel_weights.edgeweights").getChannel());
            clModel.model.loadParameters(getActivity().openFileInput("clmodel_weights.edgeweights").getChannel());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {


        CameraFragmentBinding dataBinding =
                DataBindingUtil.inflate(inflater, R.layout.camera_fragment, container, false);
        dataBinding.setLifecycleOwner(getViewLifecycleOwner());
        dataBinding.setVm(mViewModel);
        View rootView = dataBinding.getRoot();

        // Setting listeners for calss button clicks (sample additions)
        for (int buttonId : new int[]{
                R.id.class_btn_1, R.id.class_btn_2, R.id.class_btn_3, R.id.class_btn_4}) {
            rootView.findViewById(buttonId).setOnClickListener(onAddSampleClickListener);
        }

        // Setting capture modes based on Chip selection (inference vs capture)
        ChipGroup chipGroup = (ChipGroup) rootView.findViewById(R.id.mode_chip_group);
        if (mViewModel.getCaptureMode().getValue()) {
            ((Chip) rootView.findViewById(R.id.capture_mode_chip)).setChecked(true);
        } else if (mViewModel.getContinualLearningMode().getValue()) {
            ((Chip) rootView.findViewById(R.id.cl_inference_mode_chip)).setChecked(true);
        } else {
            ((Chip) rootView.findViewById(R.id.inference_mode_chip)).setChecked(true);
        }

        chipGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.capture_mode_chip) {
                mViewModel.setCaptureMode(true);
                mViewModel.setContinualLearningMode(false);
            } else if (checkedId == R.id.inference_mode_chip) {
                mViewModel.setCaptureMode(false);
                mViewModel.setContinualLearningMode(false);
            } else if (checkedId == R.id.cl_inference_mode_chip) {
                mViewModel.setCaptureMode(false);
                mViewModel.setContinualLearningMode(true);
            }
        });

        return dataBinding.getRoot();
    }

    @Override
    public void onViewCreated(View view, Bundle bundle) {
        super.onViewCreated(view, bundle);

        viewFinder = getActivity().findViewById(R.id.view_finder);
        viewFinder.post(this::startCamera);

        restoreModelVisual(mViewModel,null);

        // Resetting both models on button click
        Button btnResetModels = getActivity().findViewById(R.id.btn_reset_models);
        btnResetModels.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String scenario = "default";
                databaseHelper.emptyReplayBuffer((scenario == null) ? "default" : scenario);
                databaseHelper.emptyTrainingSamples((scenario == null) ? "default" : scenario);
                databaseHelper.emptyClassButtonImages();
                clModel.model.trainingSamples.clear();
                tlModel.model.trainingSamples.clear();
                mViewModel.resetView();
                try {
                    clModel.resetModelWeights(getActivity().openFileOutput("clmodel_weights.edgeweights", Context.MODE_PRIVATE).getChannel());
                    tlModel.resetModelWeights(getActivity().openFileOutput("tlmodel_weights.edgeweights", Context.MODE_PRIVATE).getChannel());
                    tlModel.model.loadParameters(getActivity().openFileInput("tlmodel_weights.edgeweights").getChannel());
                    clModel.model.loadParameters(getActivity().openFileInput("clmodel_weights.edgeweights").getChannel());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                getActivity().finish();
                getActivity().startActivity(getActivity().getIntent());
            }
        });
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);

        mViewModel.getTrainingState()
                .observe(
                        getViewLifecycleOwner(),
                        trainingState -> {
                            switch (trainingState) {
                                case STARTED:
                                    tlModel.enableTraining(((epoch, loss) -> mViewModel.setLastLoss(loss)));
                                    clModel.enableTraining(((epoch, loss) -> mViewModel.setLastLossCL(loss)));

                                    if (!mViewModel.getInferenceSnackbarWasDisplayed().getValue()) {
                                        Snackbar.make(
                                                getActivity().findViewById(R.id.classes_bar),
                                                "You can switch back to inference mode now.",
                                                Snackbar.LENGTH_LONG)
                                                .show();
                                        mViewModel.markInferenceSnackbarWasCalled();
                                    }
                                    break;
                                case PAUSED:
                                    tlModel.disableTraining();
                                    clModel.disableTraining();

                                    // Recording the scenario
//                                    int replay_1 = (clModel.replayBuffer.containsKey("1") ? clModel.replayBuffer.get("1").size() : 0);
//                                    int replay_2 = (clModel.replayBuffer.containsKey("2") ? clModel.replayBuffer.get("2").size() : 0);
//                                    int replay_3 = (clModel.replayBuffer.containsKey("3") ? clModel.replayBuffer.get("3").size() : 0);
//                                    int replay_4 = (clModel.replayBuffer.containsKey("4") ? clModel.replayBuffer.get("4").size() : 0);
//                                    databaseHelper.insertScenario("default",trainingIteration,
//                                            tlModel.model.trainingSamples.size(),
//                                            replay_1,replay_2,replay_3,replay_4);

                                    try {
                                        tlModel.model.saveParameters(getActivity().openFileOutput("tlmodel_weights.edgeweights", Context.MODE_PRIVATE).getChannel());
                                        clModel.model.saveParameters(getActivity().openFileOutput("clmodel_weights.edgeweights", Context.MODE_PRIVATE).getChannel());
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }


                                    clModel.model.trainingSamples.clear(); // Clearing when paused - NOT IDEAL
                                    tlModel.model.trainingSamples.clear();
                                    mViewModel.setPrevIterationTotalSamples(mViewModel.getTotalSamples().getValue());
                                    mViewModel.setTrainingIteration(true);
                                    System.gc(); // To collect bitmaps from memory
                                    clModel.updateReplayBufferSmart(null); // Updating replay buffer
                                    break;
                                case NOT_STARTED:
                                    break;
                            }
                        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        tlModel.close();
        clModel.close();
        tlModel = null;
        clModel = null;
    }

    /***
     * Setup a camera preview and launch an inference thread
     */
    private void startCamera() {
        viewFinderRotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
        if (viewFinderRotation == null) {
            viewFinderRotation = 0;
        }

        // Get metrics (aspect ratio) of the Screen
        DisplayMetrics metrics = new DisplayMetrics();
        viewFinder.getDisplay().getRealMetrics(metrics);
        Rational screenAspectRatio = new Rational(metrics.widthPixels, metrics.heightPixels);

        // Building a camera preview
        PreviewConfig config = new PreviewConfig.Builder()
                .setLensFacing(LENS_FACING)
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(viewFinder.getDisplay().getRotation())
                .build();

        Preview preview = new Preview(config);

        // Preview update listener - Changed from original code to make more sense semantically
        preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
            @Override
            public void onUpdated(Preview.PreviewOutput output) {
                ViewGroup parent = (ViewGroup) viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);

                viewFinder.setSurfaceTexture(output.getSurfaceTexture());

                Integer rotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
                updateTransform(rotation, output.getTextureSize(), viewFinderDimens);
            }
        });

        // Layout change listener for our view finder - Changed from original code to make more sense semantically
        viewFinder.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            @Override
            public void onLayoutChange(View v, int left, int top, int right, int bottom, int oldLeft,
                                       int oldTop, int oldRight, int oldBottom) {
                Size newViewFinderDimens = new Size(right - left, bottom - top);
                Integer rotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
                updateTransform(rotation, bufferDimens, newViewFinderDimens);
            }
        });

        // Building a thread for Inference, which include live image analysis (prep)
        HandlerThread inferenceThread = new HandlerThread("InferenceThread");
        inferenceThread.start();
        ImageAnalysisConfig analysisConfig = new ImageAnalysisConfig.Builder()
                .setLensFacing(LENS_FACING)
                .setCallbackHandler(new Handler(inferenceThread.getLooper()))
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setTargetRotation(viewFinder.getDisplay().getRotation())
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(analysisConfig);
        imageAnalysis.setAnalyzer(inferenceAanalyzer);

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }

    private final ImageAnalysis.Analyzer inferenceAanalyzer = (imageProxy, rotationDegrees) -> {
        final String imageId = UUID.randomUUID().toString();

        inferenceBenchmark.startStage(imageId, "preprocess");
        Bitmap bitmap = yuvCameraImageToBitmap(imageProxy);
        float[] rgbImage = prepareCameraImage(bitmap, rotationDegrees);
        inferenceBenchmark.endStage(imageId, "preprocess");

        // Adding samples is also handled by inference thread / use case.
        // We don't use CameraX ImageCapture since it has very high latency (~650ms on Pixel 2 XL)
        // even when using .MIN_LATENCY.
        String sampleClass = addSampleRequests.poll();
        if (sampleClass != null) {
            inferenceBenchmark.startStage(imageId, "addSample");
            try {
                tlModel.addSample(rgbImage, sampleClass).get();
                clModel.addSample(rgbImage, sampleClass).get();
                tlModel.storeTrainingSample(null); // Stores in Databases for both CL AND TL
                if (!mViewModel.getClassBtnImages().containsKey(sampleClass)) {
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            ImageView imageView = null;
                            if (sampleClass.equals("1")){
                                imageView = getActivity().findViewById(R.id.class_img_1);
                            }else if (sampleClass.equals("2")){
                                imageView = getActivity().findViewById(R.id.class_img_2);
                            }else if (sampleClass.equals("3")){
                                imageView = getActivity().findViewById(R.id.class_img_3);
                            }else{
                                imageView = getActivity().findViewById(R.id.class_img_4);
                            }
                            mViewModel.updateClassBtnImage(imageView,bitmap);
                            databaseHelper.insertClassButtonImage(ImageUtils.bitmapToByteArray(bitmap),sampleClass);
                        }
                    });
                    mViewModel.getClassBtnImages().put(sampleClass,bitmap);
                }
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to add sample to model", e.getCause());
            } catch (InterruptedException e) {
                // no-op
            }

            mViewModel.increaseNumSamples(sampleClass);
            inferenceBenchmark.endStage(imageId, "addSample");

        } else {
            // We don't perform inference when adding samples, since we should be in capture mode
            // at the time, so the inference results are not actually displayed.
            inferenceBenchmark.startStage(imageId, "predict");

            // Predicting using CL or TL models
            TransferLearningModel.Prediction[] predictions = null;
            if (!mViewModel.getContinualLearningMode().getValue()) {
                // Transfer Learning
                predictions = tlModel.predict(rgbImage);
            } else {
                // Continual Learning
                predictions = clModel.predict(rgbImage);
            }

            if (predictions == null) {
                return;
            }
            inferenceBenchmark.endStage(imageId, "predict");

            for (TransferLearningModel.Prediction prediction : predictions) {
                mViewModel.setConfidence(prediction.getClassName(), prediction.getConfidence());
            }
        }
        inferenceBenchmark.finish(imageId);
    };

    public final View.OnClickListener onAddSampleClickListener = view -> {
        String className;
        if (view.getId() == R.id.class_btn_1) {
            className = "1";
        } else if (view.getId() == R.id.class_btn_2) {
            className = "2";
        } else if (view.getId() == R.id.class_btn_3) {
            className = "3";
        } else if (view.getId() == R.id.class_btn_4) {
            className = "4";
        } else {
            throw new RuntimeException("Listener called for unexpected view");
        }
        addSampleRequests.add(className);
    };

    /**
     * BINDING ADAPTERS - For data binding in the view (live data changes)
     **/

    @BindingAdapter({"captureMode", "inferenceText", "captureText"})
    public static void setClassSubtitleText(TextView view, boolean captureMode, Float inferenceText, Integer captureText) {
        if (captureMode) {
            view.setText(captureText != null ? Integer.toString(captureText) : "0");
        } else {
            view.setText(String.format(Locale.getDefault(), "%.2f", inferenceText != null ? inferenceText : 0.f));
        }
    }

    @BindingAdapter({"android:visibility"})
    public static void setViewVisibility(View view, boolean visible) {
        view.setVisibility(visible ? View.VISIBLE : View.GONE);
    }

    // This method simply adds a rectangle background the class that is currently "in use"
    @BindingAdapter({"highlight"})
    public static void setClassButtonHighlight(View view, boolean highlight) {
        int drawableId;
        if (highlight) {
            drawableId = R.drawable.btn_default_highlight;
        } else {
            drawableId = R.drawable.btn_default;
        }
        view.setBackground(view.getContext().getDrawable(drawableId));
    }

    /** Extra Utilities - SHOULD MOVE TO INDEPENDENT FILE - START **/

    /***
     * Returns the rotation of the surface in certain points
     * @param display
     * @return
     */
    private static Integer getDisplaySurfaceRotation(Display display) {
        if (display == null) {
            return null;
        }

        switch (display.getRotation()) {
            case Surface.ROTATION_0:
                return 0;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_270:
                return 270;
            default:
                return null;
        }
    }

    /**
     * Fit the camera preview into [viewFinder].
     *
     * @param rotation            view finder rotation.
     * @param newBufferDimens     camera preview dimensions.
     * @param newViewFinderDimens view finder dimensions.
     */
    private void updateTransform(Integer rotation, Size newBufferDimens, Size newViewFinderDimens) {
        // Nothing has changed in the view so there is no need to proceed
        if (Objects.equals(rotation, viewFinderRotation)
                && Objects.equals(newBufferDimens, bufferDimens)
                && Objects.equals(newViewFinderDimens, viewFinderDimens)) {
            return;
        }

        // Updating Rotation
        if (rotation == null) {
            return;
        } else {
            viewFinderRotation = rotation;
        }

        // Updating buffer dimensions
        if (newBufferDimens.getWidth() == 0 || newBufferDimens.getHeight() == 0) {
            return;
        } else {
            bufferDimens = newBufferDimens;
        }

        // Updating viewfinder dimensions
        if (newViewFinderDimens.getWidth() == 0 || newViewFinderDimens.getHeight() == 0) {
            return;
        } else {
            viewFinderDimens = newViewFinderDimens;
        }

        Log.d(TAG, String.format("Applying output transformation.\n"
                + "View finder size: %s.\n"
                + "Preview output size: %s\n"
                + "View finder rotation: %s\n", viewFinderDimens, bufferDimens, viewFinderRotation));

        Matrix matrix = new Matrix();
        float centerX = viewFinderDimens.getWidth() / 2f;
        float centerY = viewFinderDimens.getHeight() / 2f;
        matrix.postRotate(-viewFinderRotation.floatValue(), centerX, centerY);

        float bufferRatio = bufferDimens.getHeight() / (float) bufferDimens.getWidth();

        int scaledWidth;
        int scaledHeight;
        if (viewFinderDimens.getWidth() > viewFinderDimens.getHeight()) {
            scaledHeight = viewFinderDimens.getWidth();
            scaledWidth = Math.round(viewFinderDimens.getWidth() * bufferRatio);
        } else {
            scaledHeight = viewFinderDimens.getHeight();
            scaledWidth = Math.round(viewFinderDimens.getHeight() * bufferRatio);
        }

        float xScale = scaledWidth / (float) viewFinderDimens.getWidth();
        float yScale = scaledHeight / (float) viewFinderDimens.getHeight();
        matrix.preScale(xScale, yScale, centerX, centerY);

        viewFinder.setTransform(matrix);
    }

    /***
     * Converts camera's YUV photo into a Bitmap for processing
     * @param imageProxy
     * @return
     */
    private static Bitmap yuvCameraImageToBitmap(ImageProxy imageProxy) {
        if (imageProxy.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException(
                    "Expected a YUV420 image, but got " + imageProxy.getFormat());
        }

        ImageProxy.PlaneProxy yPlane = imageProxy.getPlanes()[0];
        ImageProxy.PlaneProxy uPlane = imageProxy.getPlanes()[1];

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        byte[][] yuvBytes = new byte[3][];
        int[] argbArray = new int[width * height];
        for (int i = 0; i < imageProxy.getPlanes().length; i++) {
            final ByteBuffer buffer = imageProxy.getPlanes()[i].getBuffer();
            yuvBytes[i] = new byte[buffer.capacity()];
            buffer.get(yuvBytes[i]);
        }

        ImageUtils.convertYUV420ToARGB8888(
                yuvBytes[0],
                yuvBytes[1],
                yuvBytes[2],
                width,
                height,
                yPlane.getRowStride(),
                uPlane.getRowStride(),
                uPlane.getPixelStride(),
                argbArray);

        return Bitmap.createBitmap(argbArray, width, height, Bitmap.Config.ARGB_8888);
    }

    /**
     * Normalizes a camera image to [0; 1], cropping it
     * to size expected by the model and adjusting for camera rotation.
     */
    private static float[] prepareCameraImage(Bitmap bitmap, int rotationDegrees) {
        int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;

        Bitmap paddedBitmap = padToSquare(bitmap);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(
                paddedBitmap, modelImageSize, modelImageSize, true);

        Matrix rotationMatrix = new Matrix();
        rotationMatrix.postRotate(rotationDegrees);
        Bitmap rotatedBitmap = Bitmap.createBitmap(
                scaledBitmap, 0, 0, modelImageSize, modelImageSize, rotationMatrix, false);

        float[] normalizedRgb = new float[modelImageSize * modelImageSize * 3];
        int nextIdx = 0;
        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = rotatedBitmap.getPixel(x, y);

                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.f);

                normalizedRgb[nextIdx++] = r;
                normalizedRgb[nextIdx++] = g;
                normalizedRgb[nextIdx++] = b;
            }
        }
        return normalizedRgb;
    }

    /**
     * I think it turns a bitmap image into a square?
     *
     * @param source
     * @return
     */
    private static Bitmap padToSquare(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();

        int paddingX = width < height ? (height - width) / 2 : 0;
        int paddingY = height < width ? (width - height) / 2 : 0;
        Bitmap paddedBitmap = Bitmap.createBitmap(
                width + 2 * paddingX, height + 2 * paddingY, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(paddedBitmap);
        canvas.drawARGB(0xFF, 0xFF, 0xFF, 0xFF);
        canvas.drawBitmap(source, paddingX, paddingY, null);
        return paddedBitmap;
    }

    // We are storing the samples themselves in DB for later development
    public void restoreModelVisual(CameraFragmentViewModel cameraViewModel, String scenario){
        Cursor res = databaseHelper.getTrainingSamples((scenario == null) ? "default" : scenario);
        //Toast.makeText(getActivity(), "AEL RESTORED SAMPLES: " + res.getCount(), Toast.LENGTH_SHORT).show();
        if (res.getCount() != 0) {
            while (res.moveToNext())
            {
                String className = res.getString(1);
                cameraViewModel.increaseNumVisualSamples(className);
            }
        } else {
            System.out.println("AEL: DEN DOYLEUEI RESTORED");
        }

        res = databaseHelper.getClassButtonImages();
        if (res.getCount() != 0){
            while (res.moveToNext()){
                String sampleClass = res.getString(1);
                System.out.println("AEL CLASS "+sampleClass);
                byte[] blob = res.getBlob(2);
                Bitmap bitmap = BitmapFactory.decodeByteArray(blob, 0, blob.length);
                if (!mViewModel.getClassBtnImages().containsKey(sampleClass)) {
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            ImageView imageView = null;
                            if (sampleClass.equals("1")) {
                                imageView = getActivity().findViewById(R.id.class_img_1);
                            } else if (sampleClass.equals("2")) {
                                imageView = getActivity().findViewById(R.id.class_img_2);
                            } else if (sampleClass.equals("3")) {
                                imageView = getActivity().findViewById(R.id.class_img_3);
                            } else {
                                imageView = getActivity().findViewById(R.id.class_img_4);
                            }
                            if (imageView!=null) {
                                mViewModel.updateClassBtnImage(imageView, bitmap);
                            }else{
                                System.out.println("AEL einai null toimg");
                            }
                        }
                    });
                    mViewModel.getClassBtnImages().put(sampleClass, bitmap);
                }
            }
        }
    }

    /** Extra Utilities - SHOULD MOVE TO INDEPENDENT FILE - END **/

}