
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

/**
 * This is a project of a safety helmet detection using YOLOv2 architecture.
 * This project illustrate a model training process with transfer learning approach by fine tuning the last few layers of a YOLOv2 pretrained model
 * This model have 2 classes and able to detect safty helmet and head in images and webcam.
 */
//
public class PPEDetector {
    private static final Logger log = LoggerFactory.getLogger(PPEDetector.class);
    private static int seed = 11;
    private static double detectionThreshold = 0.7;
    private static int nBoxes = 6;
    private static double lambdaNoObj = 0.5; //(if there is lower value it learn better, classification task)
    private static double lambdaCoord = 5.0; //(if there is higher value it learn better, localization task)
    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}, {1.5, 3}};

    private static int batchSize = 2;
    private static int nEpochs = 20;
    private static double learningRate = 1e-4;
    private static int nClasses = 3;
    private static List<String> labels;

    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/PPEDetector.zip");
    private static ComputationGraph model;
    private static Frame frame = null;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(255, 255, 0);
   // private static final Scalar BLUE = RGB(0,0,255);
    private static Scalar[] colormap = {GREEN, YELLOW};
    private static String labeltext = null;

    private static Random rng = new Random(seed);
    // private static String dataDir;
    //  private static String downloadLink;
    //  private static Path trainDir, testDir;
    //  private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;

    public static void main(String[] args) throws Exception {

        //        STEP 1 : Create iterators
        File trainDir = new ClassPathResource("ppe/train").getFile();
        File testDir = new ClassPathResource("ppe/test").getFile();
        log.info("Load Data...");


        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);


        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new LabelImgXmlLabelProvider(trainDir));
        recordReaderTrain.initialize(trainData);
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
                gridHeight, gridWidth, new LabelImgXmlLabelProvider(testDir));
        recordReaderTest.initialize(testData);


        //Since ObjectDetectionReader is regression, we need to specify it here,
        RecordReaderDataSetIterator trainIter = new RecordReaderDataSetIterator(recordReaderTrain,
                batchSize, 1, 1, true);
        trainIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest,
                1, 1, 1, true);
        testIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        labels = trainIter.getLabels();


        System.out.println(Arrays.toString(labels.toArray()));

        //        If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {
            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
            System.out.println(model.summary());
        } else
           {
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorBoxes);
            //     STEP 2 : Train the model using Transfer Learning
            //     STEP 2.1: Transfer Learning steps - Load YOLO2 prebuilt model.
            log.info("Build model...");
            ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            //     STEP 2.2: Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            //     STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getComputationGraph(pretrained, priors, fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(
                    yoloheight,
                    yolowidth,
                    nClasses)));

            //     STEP 2.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1; i < nEpochs + 1; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }
        //     STEP 3: Evaluate the model's accuracy by using the test iterator.
        OfflineValidationWithTestDataset(testIter);
        //     STEP 4: Inference the model and process the webcam stream and make predictions.
       // doInference();
    }

    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))  //nBoxes =6, nClasses=2
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }

    //    Evaluate visually the performance of the trained object detection model
    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            YoloUtils.nms(objs, 0.4);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));
            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();
    }

    // Stream video frames from Webcam and run them through YOLOv2 model and get predictions
 /*   private static void doInference() {

        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(
                FruitDataSetIterator.yolowidth,
                FruitDataSetIterator.yoloheight,
                3,
                new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("Object Detection");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

        while (true) {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(FruitDataSetIterator.yolowidth, FruitDataSetIterator.yoloheight));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
                            YoloUtils.nms(objs, 0.4);
                            rawImage = drawResults(objs, rawImage, w, h);
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            KeyEvent t = null;
            try {
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
    }

  */

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            //Draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            //Display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
        }
        return mat;
    }
}



