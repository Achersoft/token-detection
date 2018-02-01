
import com.achersoft.tokens.DetectTokens;
import com.achersoft.tokens.Recognizer;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.imageio.ImageIO;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author shaun
 */
public class test {
    public static void main(String[] args) throws IOException, Exception {
        DetectTokens detector = new DetectTokens(0.20F);
        
        List<BufferedImage> images = detector.findTokens(ImageIO.read(new File("C:\\Users\\shaun\\Repositories\\tensorflow-for-poets-2\\tf_files\\test\\1.jpg")));
        System.err.println(images.size());
        
        Recognizer recon = new Recognizer();
        System.err.println(recon.identify(images));
    }
}
