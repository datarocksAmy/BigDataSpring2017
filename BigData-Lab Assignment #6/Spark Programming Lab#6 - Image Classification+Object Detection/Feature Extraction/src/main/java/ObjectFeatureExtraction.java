import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;

import java.io.*;

/**
 * =====================================================================
 * CS5542 Big Data Analytics & Application Lab
 * Assignment #6 - Object Feature Extraction
 * Obtain Feature Extraction for each image category seperately
 * #20 Chia-Hui Amy Lin
 * =====================================================================
 */
public class ObjectFeatureExtraction {
    public static void main(String args[]) throws IOException {
        String inputFolder = "data/";
        String inputImage = "chameleon2.jpg";  // Manually doing to test the images; What kind of image details/features you want to get
        String outputFolder = "output/";
        String[] IMAGE_CATEGORIES = {"Sea", "PolarBear", "Fish", "Peacock", "Chameleon"};

        int input_class = 4; // Change to the index from IMAGE_CATEGORIES of the category you want to extract the feature.
        MBFImage mbfImage = ImageUtilities.readMBF(new File(inputFolder + inputImage));
        DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();
        LocalFeatureList<Keypoint> features = doGSIFTEngine.findFeatures(mbfImage.flatten());
        FileWriter fw = new FileWriter(outputFolder + IMAGE_CATEGORIES[input_class] + ".txt");
        BufferedWriter bw = new BufferedWriter(fw);
        for (int i = 0; i < features.size(); i++) {
            double c[] = features.get(i).getFeatureVector().asDoubleVector();
            bw.write(input_class + ",");
            for (int j = 0; j < c.length; j++) {
                bw.write(c[j] + " ");
            }
            bw.newLine();
        }
        bw.close();
    }
}
