import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.util.ArchiveUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;

public class MyDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(MyDataSetIterator.class);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 10;

    //Images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    //Random number generator
    private static final Random rng  = new Random(123);

    private static String dataDir;


    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public MyDataSetIterator() throws IOException {
    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        if (training && transform != null){
            recordReader.initialize(split,transform);
        }else{
            recordReader.initialize(split);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, false);
    }

    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransform) throws IOException {
        transform=imageTransform;
        setup(batchSizeArg,trainPerc);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"), //dir: C:\Users\User
                Helper.getPropValues("dl4j_home.data") //dir: C:\Users\User\.deeplearning4j\data
        ).toString();

        File parentDir = new File(Paths.get(dataDir,"clothes").toString()); //dir: C:\Users\User\.deeplearning4j\data\clothes

        if(!parentDir.exists()){
            System.out.println("Dude data is not here");
        }
        batchSize = batchSizeArg;

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }


}

