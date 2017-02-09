/**
 * =====================================================================
 * CS5542 Big Data Analytics & ApplicationLab
 * Assignment #3 - Image Annotation (Extra Example)
 * A simple image annotation example
 * #20 Chia-Hui Amy Lin
 * =====================================================================
 */

import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.List;


public class ClarifaiExample {
    public static void main(String[] args) throws IOException {

        // Connect the Clarifai API server by using your own API key and access code to get the token.
        final ClarifaiClient client = new ClarifaiBuilder("eTJs-4TeKoI7oj9KxDvAtXIAcMlq_dGA5q9O-Leo", "Z8AvX0xfORRPeDT6-rzTIeeQP06MBt39bOG-26Jz")
                .client(new OkHttpClient()) // OPTIONAL. Allows customization of OkHttp by the user
                .buildSync(); // or use .build() to get a Future<ClarifaiClient>
        client.getToken();

        // Read in the animal.jpg image and start the prediction process of this image.
        ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                .withInputs(
                        ClarifaiInput.forImage(ClarifaiImage.of(new File("input/animal.jpg")))
                )
                .executeSync();
        List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();
        if (predictions.isEmpty()) {
            System.out.println("No Predictions");
        }
        else{
            MBFImage image = ImageUtilities.readMBF(new File("input/animal.jpg"));
            int x = image.getWidth();
            int y = image.getHeight();


            List<Concept> data = predictions.get(0).data();
            for (int i = 0; i < data.size(); i++) {
                System.out.println(data.get(i).name() + " - " + data.get(i).value());
                image.drawText(data.get(i).name(), (int)Math.floor(Math.random()*x), (int) Math.floor(Math.random()*y), HersheyFont.ASTROLOGY, 20, RGBColour.RED);
            }

            // Display the possible info on the image to the user
            DisplayUtilities.displayName(image, "videoFrames");

        }

    }
}
