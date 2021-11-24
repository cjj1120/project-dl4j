import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class main {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(main.class);

    private static final int outputNum = 3;
    private static final int seed = 123;
    private static final int trainPerc = 80;
    private static final int batchSize = 16;
    private static File modelFilename = new File(System.getProperty("user.home"), ".deeplearning4j/generated-models/test0--vgg16-model.zip");
    private static ComputationGraph vgg16Transfer;
    private static List<String> labels;

    public static void main(String[] args) throws IOException, IllegalAccessException {
        //Load model if it exists
        if (modelFilename.exists()) {
            log.info("Load model...");
            vgg16Transfer = ModelSerializer.restoreComputationGraph(modelFilename);
            //testImage();
            log.info("Model loaded");
        } else {

        log.info("Model building... ...");
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());


        // ORI MODEL
//        FineTuneConfiguration fineTuneCOnf = new FineTuneConfiguration.Builder()
//                .updater(new Nesterovs(5e-5))
//                .seed(seed)
//                .build();
//
//        //
//         vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
//                .fineTuneConfiguration(fineTuneCOnf)
//                .setFeatureExtractor("fc2")
//                .removeVertexKeepConnections("predictions")
//                 //.removeVertexKeepConnections("fc2")
//                 //Example of adding dense layer with REST neural network
////                 .addLayer("fc2",
////                         new DenseLayer.Builder()
////                                 .activation(Activation.RELU)
////                                 .nOut(4096)
////                                 .build(),
////                         "fc1") //add in a new dense layer
//
//                .addLayer("predictions",
//                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                                .nIn(4096)
//                                .nOut(outputNum)
//                                .weightInit(WeightInit.XAVIER)
//                                .activation(Activation.SOFTMAX).build(),
//                        "fc2")
//                .build();


        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-4))
//                .dropOut(0.2)
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("block5_pool") //"block5_pool" and below are frozen
                .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
                .removeVertexAndConnections("predictions") //remove the final vertex and it's connections

                .addLayer("fc3",new DenseLayer
                        .Builder().activation(Activation.RELU).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
                .addLayer("newpredictions",new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256)
                        .nOut(outputNum)
                        .build(),"fc3") //add in a final output dense layer,
                .setOutputs("newpredictions") // specify the new output we defined above
                .build();



        log.info(vgg16Transfer.summary());

        }


        //
        MyDataSetIterator.setup(batchSize, trainPerc);
        DataSetIterator trainIter = MyDataSetIterator.trainIterator();
        DataSetIterator testIter = MyDataSetIterator.testIterator();
            //Print labels
        labels = trainIter.getLabels();
        System.out.println(Arrays.toString(labels.toArray()));

        //
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //Damn I miss out this part.. see if it makes a difference
        vgg16Transfer.setListeners(
                new StatsListener(storage),
                new ScoreIterationListener(1),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));


        //
        int iter = 0;
        while (trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter " + iter + "...");
                Evaluation eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        log.info("Model build complete");
        ModelSerializer.writeModel(vgg16Transfer, modelFilename, true);
        log.info("Model saved");

//        Evaluation eval1 = vgg16Transfer.evaluate(trainIter);
//        System.out.println(eval1.stats());
        Evaluation eval = vgg16Transfer.evaluate(testIter);
        System.out.println(eval.stats());

        log.info("Program End");
    }



//    private static void testImage() throws Exception {
//        String testImagePATH = "D:\\Desktop\\facial emotion\\m.jpg";    //change path to own image dir
//        File file = new File(testImagePATH);
//        System.out.println(String.format("You are using this image file located at %s", testImagePATH));
//        NativeImageLoader nil = new NativeImageLoader(48, 48, 1);
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//
//
//        INDArray image = nil.asMatrix(file);
//        scaler.transform(image);
//
//
//        Mat opencvMat = imread(testImagePATH);
//        INDArray outputs = model.output(image);
//        INDArray op = Nd4j.argMax(outputs, 1);
//
//        int ans = op.getInt(0);
//
//        if (ans == 0) {
//            log.info("Emotion : Angry");
//        }
//        if (ans == 1) {
//            log.info("Emotion : Happy");
//        }
//        if (ans == 2) {
//            log.info("Emotion : Neutral");
//        }
//
//        log.info("Label:         " + Nd4j.argMax(outputs, 1));
//        log.info("Probabilities: " + outputs.toString());
//
//        imshow("Input Image", opencvMat);
//
//        if (waitKey(0) == 27) {
//            destroyAllWindows();
//        }
//
//    }



    //SECOND TEST EXAMPLE height,width = 150; channels=3, labels is declared above
//    private static void TestWithSingleImage() throws IOException{
//
//        File my_image= new File("C:\\Users\\choowilson\\Desktop\\test\\rock\\rock.jpg");
//        log.info("You are using this image file located at {}", my_image );
//        NativeImageLoader loader = new NativeImageLoader(height,width, channels);
//        INDArray image = loader.asMatrix(my_image);
//        INDArray output = model.output(image);
//        log.info("Labels: {}", Arrays.toString(labels.toArray()));
//        log.info("Confidence Level: {}",output);
//        log.info("Predicted class: {}",labels.toArray()[model.predict(image)[0]]);
//    }
}

