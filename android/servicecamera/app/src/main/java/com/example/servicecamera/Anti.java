package com.example.servicecamera;

import static androidx.constraintlayout.helper.widget.MotionEffect.TAG;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;

import android.widget.ImageView;

import com.example.namespace.R;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

public class Anti extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Bitmap bitmap1 = null;
        Bitmap bitmap2 = null;
        Bitmap bitmap3 = null;
        Bitmap bitmap4 = null;
        Bitmap a= null;
        Bitmap b= null;
        Module module = null;
        Module antimodule1 = null;
        Module antimodule2 = null;
        Mat mRgba = new Mat();

        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg
            bitmap1 = BitmapFactory.decodeStream(getAssets().open("face-1.jpg"));
            bitmap2 = BitmapFactory.decodeStream(getAssets().open("face.jpg"));
            bitmap3 = BitmapFactory.decodeStream(getAssets().open("image_F1.jpg"));

            //bitmap1 = Bitmap.createScaledBitmap(bitmap1, 256, 256, false);
            bitmap1 = BitmapUtil.resizeBitmap(bitmap1, 256);
            bitmap1 = BitmapUtil.cropCenterBitmap(bitmap1,224, 224);

            bitmap2 = BitmapUtil.resizeBitmap(bitmap2, 256);
            bitmap2 = BitmapUtil.cropCenterBitmap(bitmap2,224, 224);

            bitmap3 = Bitmap.createScaledBitmap(bitmap3, 80, 80, true);
//            bitmap3 = BitmapUtil.resizeBitmap(bitmap3, 256);
//            bitmap3 = BitmapUtil.cropCenterBitmap(bitmap3,224, 224);
//            bitmap3 = resizeBitmapImage(bitmap3,80);

            Bitmap bmp32 = bitmap3.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(bmp32, mRgba);


            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            System.out.println("start load ptl file");
            //module = LiteModuleLoader.load(assetFilePath(this, "jit_model.pt"));
            antimodule1 = LiteModuleLoader.load(assetFilePath(this, "anti.pt"));
            antimodule2 = LiteModuleLoader.load(assetFilePath(this, "anti2.pt"));
            System.out.println("end load ptl file");
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }
        //crop
        String proto = null;
        try {
            proto = assetFilePath(this,"deploy.prototxt");
        } catch (IOException e) {
            e.printStackTrace();
        }
        String weights = null;
        try {
            weights = assetFilePath(this, "Widerface-RetinaFace.caffemodel");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Net net = Dnn.readNetFromCaffe(proto, weights);
        Log.i(TAG, "Network loaded successfully");


        Size sz = new Size(480,640);
        Mat resizedimage = new Mat();

        Imgproc.resize( mRgba, resizedimage, sz );

        int height2 = resizedimage.cols();
        int width2 = resizedimage.rows();
        float aspect_ratio = (float) width2 / height2;

        if (height2 * width2 >= 192 * 192) {
            Size sz2 = new Size((int) (192 * Math.sqrt(aspect_ratio)), (int) (192 / Math.sqrt(aspect_ratio)));
            Mat resizedimage2 = new Mat();
            Imgproc.resize(resizedimage, resizedimage2, sz2);
            resizedimage = resizedimage2;
        }
        Imgproc.cvtColor(resizedimage, resizedimage, Imgproc.COLOR_RGBA2RGB);
        Mat blob = Dnn.blobFromImage(resizedimage, 1,
                new Size(resizedimage.rows(), resizedimage.cols()),
                new Scalar(104, 117, 123));

        // Mat dst_mat = new Mat();
//        Imgproc.cvtColor(blob, blob, Imgproc.COLOR_RGBA2BGR);
        net.setInput(blob);
        Mat detections = net.forward();
        detections = detections.reshape(0, detections.size(2));
        Mat dd = detections.col(2);
        float[] arrdd = new float[dd.rows()*dd.cols()];
        dd.get(0, 0, arrdd);
        int argmax = argmax(arrdd);
        Mat test = Mat.ones(detections.rows(),detections.cols(), CvType.CV_32FC1);
        double[] left = detections.mul(test, width2).get(argmax,3);
        double[] top = detections.mul(test, width2).get(argmax,5);
        double[] right = detections.mul(test, height2).get(argmax,4);
        double[] bottom = detections.mul(test, height2).get(argmax,6);

        int x = (int) left[0];
        int y = (int) top[0];
        int box_w = (int) (right[0] - left[0] + 1);
        int box_h = (int) (bottom[0] - top[0 + 1]);

        // Test anti spoofing
        float[] NORM_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
        float[] NORM_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};
        Tensor antiTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap3,
                NORM_MEAN_RGB, NORM_STD_RGB);

        float[] tmp2 = antiTensor.getDataAsFloatArray();
        for(int i=0; i<tmp2.length;i++)
        {
            tmp2[i] *= 255.0;
        }
        antiTensor = Tensor.fromBlob(tmp2, antiTensor.shape());
        tmp2 = antiTensor.getDataAsFloatArray();


        Tensor outputAntiTensor1 = antimodule1.forward(IValue.from(antiTensor)).toTensor();
        Tensor outputAntiTensor2 = antimodule2.forward(IValue.from(antiTensor)).toTensor();


        float[] anti_tmp1 = outputAntiTensor1.getDataAsFloatArray();
        //double result = softmax(3, convertFloatsToDoubles(tmp));

        float[] anti_tmp2 = outputAntiTensor2.getDataAsFloatArray();
        //double result = softmax(3, convertFloatsToDoubles(tmp));

        float[] sum_anti = new float[] {0.0f, 0.0f, 0.0f};
        for(int i=0;i<anti_tmp1.length;i++)
        {
            sum_anti[i] = anti_tmp1[i] + anti_tmp2[i];
        }
        //double result = softmax(3, convertFloatsToDoubles(tmp));
        int max_tmp1= argmax(sum_anti);
        String test_tmp = "";
    }

    public static int argmax(float[] a) {
        float re = Float.MIN_VALUE;
        int arg = -1;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > re) {
                re = a[i];
                arg = i;
            }
        }
        return arg;
    }


    public double softmax(double input, double[] neuronValues) {
        double total = Arrays.stream(neuronValues).map(Math::exp).sum();
        return Math.exp(input) / total;
    }

    public static double[] convertFloatsToDoubles(float[] input)
    {
        if (input == null)
        {
            return null; // Or throw an exception - your choice
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i];
        }
        return output;
    }

    public static float cosineSimilarity(Tensor a, Tensor b) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        float[] vectorA = a.getDataAsFloatArray();
        float[] vectorB = b.getDataAsFloatArray();

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public Bitmap resizeBitmapImage(Bitmap source, int maxResolution)
    {
        int width = source.getWidth();
        int height = source.getHeight();
        int newWidth = width;
        int newHeight = height;
        float rate = 0.0f;

        if(width > height)
        {
            if(maxResolution < width)
            {
                rate = maxResolution / (float) width;
                newHeight = (int) (height * rate);
                newWidth = maxResolution;
            }
        }
        else
        {
            if(maxResolution < height)
            {
                rate = maxResolution / (float) height;
                newWidth = (int) (width * rate);
                newHeight = maxResolution;
            }
        }
        return Bitmap.createScaledBitmap(source, newWidth, newHeight, true);
    }
}