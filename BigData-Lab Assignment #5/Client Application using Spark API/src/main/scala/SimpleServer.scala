/**
  * =====================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #5 - Client Application using Spark API
  * Using JetBrains Extension on Google to connect the Spark Program for image recognition.
  * User defined dataset. --- bubble tea, chocolate, coffee, mocha tea
  * #20 Chia-Hui Amy Lin
  * =====================================================================
  */
import java.io.{File, ByteArrayInputStream}
import java.nio.file.{Files, Paths}
import javax.imageio.{ImageWriteParam, IIOImage, ImageIO}
//import java.util.Base64
import sun.misc.BASE64Decoder;
import _root_.unfiltered.request.Body
import _root_.unfiltered.request.Path
import _root_.unfiltered.response.Ok
import _root_.unfiltered.response.ResponseString
import unfiltered.filter.Plan
import unfiltered.jetty.SocketPortBinding
import unfiltered.request._


object SimplePlan extends Plan {
  System.setProperty("hadoop.home.dir","C:\\winutils");
  def intent = {
    case req @ GET(Path("/get")) => {
      Ok ~> ResponseString(IPApp.testImage("data3/TestData/bubbletea/bubbletea2.jpg"))
    }

    case req @ POST(Path("/get_custom")) => {
      val imageByte = (new BASE64Decoder()).decodeBuffer(Body.string(req));
      val bytes = new ByteArrayInputStream(imageByte)
      val image = ImageIO.read(bytes)

      ImageIO.write(image, "png", new File("image.png"))
      Ok ~> ResponseString(IPApp.testImage("image.png"))
    }
  }
}
object SimpleServer extends App {
  val bindingIP = SocketPortBinding(host = "127.0.0.1", port = 8080)
  unfiltered.jetty.Server.portBinding(bindingIP).plan(SimplePlan).run()
}
