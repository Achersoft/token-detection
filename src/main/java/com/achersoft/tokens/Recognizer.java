package com.achersoft.tokens;

import com.google.common.io.ByteStreams;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Recognizer {
    
    private final Graph graph; 
    private final ArrayList<String> labels;
    
    public Recognizer() throws Exception {
        graph = new Graph();
        graph.importGraphDef(loadGraphDef());
        labels = loadLabels();
    }
    
    public List<String> identify(List<BufferedImage> images) throws Exception {
        Session session = new Session(graph);
        List<String> foundTokens = new ArrayList<>();
 
        images.stream().forEach((img) -> {
            try {
                Tensor<String> input = Tensors.create(toByteArrayAutoClosable(img, "jpg"));
                Tensor<Float> output = session
                        .runner()
                        .feed("DecodeJpeg/contents", input)
                        .fetch("final_result")
                        .run()
                        .get(0)
                        .expect(Float.class);
                float[] probabilities = output.copyTo(new float[1][(int)output.shape()[1]])[0];
                int label = argmax(probabilities);
                if (probabilities[label] * 100.0 > 90.0)
                    foundTokens.add(labels.get(label));
                System.out.println(labels.get(label) + " is " + probabilities[label] * 100.0);
            } catch (IOException ex) {
                Logger.getLogger(Recognizer.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
        
        return foundTokens;
    }

     private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          Tensor<Float> result =
              s.runner().feed("DecodeJpeg/contents", image).fetch("final_result").run().get(0).expect(Float.class)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }
     
     
  private byte[] loadGraphDef() throws IOException {
    try (InputStream is = getClass().getResourceAsStream("/classifier/graph.pb")) {
      return ByteStreams.toByteArray(is);
    }
  }

  private ArrayList<String> loadLabels() throws IOException {
    ArrayList<String> labels = new ArrayList<>();
    String line;
    final InputStream is = getClass().getResourceAsStream("/classifier/labels.txt");
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
      while ((line = reader.readLine()) != null) {
        labels.add(line);
      }
    }
    return labels;
  }

  private static int argmax(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

    private static byte[] toByteArrayAutoClosable(BufferedImage image, String type) throws IOException {
        try (ByteArrayOutputStream out = new ByteArrayOutputStream()){
            ImageIO.write(image, type, out);
            return out.toByteArray();
        }
    }

}
