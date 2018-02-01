package com.achersoft.tokens;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

public class DetectTokens {
    
    private final SavedModelBundle model; 
    private final float tolerance; 
    
    public DetectTokens(float tolerance) {
        this.model = SavedModelBundle.load(getClass().getResource("/saved_model").getPath().replaceFirst("/", ""), "serve");
        this.tolerance = tolerance;
    }
    
    public List<BufferedImage> findTokens(BufferedImage img) throws Exception {
        Tensor<UInt8> input = makeImageTensor(img);
        List<Tensor<?>> outputs = model.session()
                .runner()
                .feed("image_tensor", input)
                .fetch("detection_scores")
                .fetch("detection_boxes")
                .run();
        
        Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
        Tensor<Float> boxesT = outputs.get(1).expect(Float.class);
        int maxObjects = (int) scoresT.shape()[1];
        float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
        float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
        List<BufferedImage> tokens = new ArrayList<>();
        
        for (int i = 0; i < scores.length; ++i) {
            if (scores[i] < tolerance) 
              continue;
            
            int yMin = Math.round(img.getHeight()*boxes[i][0]);
            int xMin = Math.round(img.getWidth()*boxes[i][1]);
            int yMax = Math.round(img.getHeight()*boxes[i][2]);
            int xMax = Math.round(img.getWidth()*boxes[i][3]);
            
            tokens.add(img.getSubimage(xMin, yMin, (xMax-xMin), (yMax-yMin)));
        }
        
        return tokens;
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    private static Tensor<UInt8> makeImageTensor(BufferedImage img) throws IOException {
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                String.format(
                    "Expected 3-byte BGR encoding in BufferedImage, found %d. This code could be made more robust",
                    img.getType()));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }
}
