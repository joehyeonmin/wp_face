package com.example.servicecamera;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import com.example.namespace.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class FdActivity2 extends CameraActivity implements CvCameraViewListener2 {

    private static final String    TAG                 = "RnD-Project";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    private ImageView homeImage, pencilImage;
    private Button button;

    private Module module = null; //얼굴 인식 모델

    private Module module_anti1 = null; //anti1 모델
    private Module module_anti2 = null; //anti2 모델

    private int FLAG = 0; // 등록 버튼 눌렀을 때 FLAG: 0 -> 1

    private String userName; // 사용자 이름 (등록 이름, 파일명)

    // private WallPadAPI Wpapi = null; // 도어락 -> Walllpad API 사용

    public Tensor outputTensor = null;
    public Tensor inputTensor = null;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity2() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        // Wpapi = new WallPadAPI(this); // Wallpad API 객체 생성

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); // 화면 켜짐 유지

        setContentView(R.layout.face_detect_surface_view);


        new Thread(new Runnable() {
            public void run() {
                mOpenCvCameraView = findViewById(R.id.fd_activity_surface_view);
                mOpenCvCameraView.post(new Runnable() {
                    public void run() {
                        mOpenCvCameraView.setCvCameraViewListener(FdActivity2.this);
                    }
                });
            }
        }).start();
        // 카메라 뷰
//        mOpenCvCameraView = findViewById(R.id.fd_activity_surface_view);
//        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
//        mOpenCvCameraView.setCvCameraViewListener(this);

        // 홈 버튼
        homeImage = findViewById(R.id.home);
        homeImage.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v){
                // 홈버튼 -> 월패드 메인 화면
                Intent intent = getPackageManager().getLaunchIntentForPackage("kr.co.icontrols.wallpadmain");
                startActivity(intent);
            }
        });

        //사용자 얼굴 feature값 (tensor 파일->.pt) 저장 경로
        File dir = new File("/mnt/sdcard/face/");
        File files[] = dir.listFiles();

        //빈 데이터 리스트 생성
        final ArrayList<String> items = new ArrayList<>();

        //.pt파일로 끝나는 파일 명 (username) 리스트에 추가
        for (int j = 0; j < files.length; j++) {
            if (files[j].getName().endsWith(".pt")) {
                items.add(files[j].getName().substring(0,files[j].getName().length()-3));
            }
        }

        // Listview Adapter
        final ArrayAdapter adapter = new ArrayAdapter(this, android.R.layout.simple_list_item_1, items);

        // Listview 생성 및 adapter 지정
        final ListView listview = (ListView) findViewById(R.id.listview);
        listview.setAdapter(adapter);

        // 사용자 이름 등록 edittext
        final EditText txtEdit = new EditText(FdActivity2.this);

        // 사용자 이름 등록 버튼
        pencilImage = findViewById(R.id.pencil);
        pencilImage.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // 사용자 이름 등록 버튼 -> 다이얼로그
                AlertDialog.Builder builder = new AlertDialog.Builder(FdActivity2.this);
                builder.setTitle("사용자 등록");
                builder.setPositiveButton("확인", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        // "사용자 등록" 다이얼로그 확인 -> 사용자 이름 등록
                        AlertDialog.Builder builder1 = new AlertDialog.Builder(FdActivity2.this);
                        builder1.setTitle("사용자 이름을 입력하세요.");
                        builder1.setView(txtEdit);
                        builder1.setPositiveButton("확인", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                // 사용자 이름 등록 -> List에 추가
                                userName = txtEdit.getText().toString();
                                Toast.makeText(getApplicationContext(),userName,Toast.LENGTH_LONG).show();
                                // name 추가
                                items.add(userName);
                                // ListView 갱신
                                adapter.notifyDataSetChanged();
                                // 사용자 얼굴 인식 -> 등록 완료
                                FLAG = 1;
                            }
                        });
                        builder1.show();
                    }
                });

                builder.setNegativeButton("취소", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        // "사용자 등록" 다이얼로그 취소 -> 다이얼로그 삭제
                        dialogInterface.cancel();
                    }
                });
                AlertDialog alertD = builder.create();
                alertD.show();
            }
        });

        // Model Load
        try {
            module = LiteModuleLoader.load(assetFilePath(this, "v2_jit_model.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets", e);
            finish();
        }

        // Anti 1 Model Load
        try {
            module_anti1 = LiteModuleLoader.load(assetFilePath(this, "anti.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets", e);
            finish();
        }

        // Anti 2 Model Load
        try {
            module_anti2 = LiteModuleLoader.load(assetFilePath(this, "anti2.pt"));
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets", e);
            finish();
        }

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected List<CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();

        Mat croppedImage = null;
        Point x = null;
        Point y = null;

        // Face Detect
        for (int i = 0; i < facesArray.length; i++) {

            int scaleX = (int) (facesArray[i].br().x - facesArray[i].tl().x) / 4;
            int scaleY = (int) (facesArray[i].br().y - facesArray[i].tl().y) / 4;

            double x1 = facesArray[i].tl().x - scaleX;
            double y1 = facesArray[i].tl().y - scaleY;
            double x2 = facesArray[i].br().x + scaleX;
            double y2 = facesArray[i].br().y + scaleY;

            if (facesArray[i].tl().x - scaleX < 0 || facesArray[i].tl().y - scaleY < 0 || x2 > mRgba.width() || y2 > mRgba.height()) {
                x = new Point(facesArray[i].tl().x, facesArray[i].tl().y);
                y = new Point(facesArray[i].br().x, facesArray[i].br().y);
                Rect roi = new Rect(x, y);
                croppedImage = new Mat(mRgba, roi);
            } else {
                x = new Point(x1, y1);
                y = new Point(x2, y2);
                Rect roi = new Rect(x, y);
                croppedImage = new Mat(mRgba, roi);
            }

            // Face detect box 그려줌
            Imgproc.rectangle(mRgba, x, y, FACE_RECT_COLOR, 3);

            // Mat -> Bitmap -> Resize -> Crop Center
            Bitmap bmp = Bitmap.createBitmap(croppedImage.cols(), croppedImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedImage, bmp);
            bmp = BitmapUtil.resizeBitmap(bmp, 256);
            bmp = BitmapUtil.cropCenterBitmap(bmp, 224, 224);

            // Anti2에 사용할 bitmap
            Bitmap bmp2 = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mRgba, bmp2);
            bmp2 = Bitmap.createScaledBitmap(bmp2, 80, 80, true);

            // Anti-spoofing
            float[] NORM_MEAN_RGB = new float[]{0.0f, 0.0f, 0.0f};
            float[] NORM_STD_RGB = new float[]{1.0f, 1.0f, 1.0f};
            Tensor antiTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp2,
                    NORM_MEAN_RGB, NORM_STD_RGB);

            float[] tmp2 = antiTensor.getDataAsFloatArray();
            for (int a = 0; a < tmp2.length; a++) {
                tmp2[a] *= 255.0;
            }
            antiTensor = Tensor.fromBlob(tmp2, antiTensor.shape());

            Tensor outputAntiTensor2 = module_anti2.forward(IValue.from(antiTensor)).toTensor();

            float[] anti_tmp2 = outputAntiTensor2.getDataAsFloatArray();

            int max_tmp1 = argmax(anti_tmp2);

            for (int a = 0; a < anti_tmp2.length; a++) {
                Log.i(TAG, "---anti_tmp2: " + anti_tmp2[a]);
            }

            Log.i(TAG, "---------max_tmp1: " + max_tmp1);
            String test = "";

            if (max_tmp1 == 0 && max_tmp1 == 2) {
                Log.i(TAG, "사진");
                break;
            } else {
                // Bitmap -> Tensor
                inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                // Tensor -> Model
                outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

                // Tensor 저장
                Map<String, Double> ht = new HashMap<String, Double>();

                // 등록된 사용자 feature 저장 경로
                File dir = new File("/mnt/sdcard/face/");
                File files[] = dir.listFiles();

                // Feature 비교 --> score
                try {
                    for (int j = 0; j < files.length; j++) {
                        if (files[j].getName().endsWith(".pt")) {
                            File file = new File(files[j].toString());

                            int size = (int) file.length();
                            byte[] bytes = new byte[size];

                            FileInputStream buf = new FileInputStream(file);
                            buf.read(bytes, 0, bytes.length);
                            buf.close();

                            float[] com_feat = byteArrayToFloatArray(bytes);
                            double score = cosine_similarity(outputTensor.getDataAsFloatArray(), com_feat);
                            String[] fileNames = files[j].getName().split("/");
                            String[] fileName = fileNames[0].split("\\.");
                            ht.put(fileName[0], score);
                            Log.i(TAG, "userName: " + userName + "score: " + score);
                        }
                    }
                } catch (RuntimeException re) {
                    Log.e(TAG, re.getMessage());
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }

                // score 비교 -> 70프로 이상의 최대값 도어락 열어줌
                Double maxVal = Collections.max(ht.values());

                for (String key : ht.keySet()) {
                    Double value = ht.get(key);
                    if (value > 0.7 && value == maxVal) {
                        Log.i(TAG, "value: " + value);
                        Imgproc.putText(mRgba, key, x, 2, 2, new Scalar(0, 255, 0));

                        Log.i(TAG, "key: " + key);
                        Toast.makeText(getApplicationContext(), key + "님이 들어왔습니다.", Toast.LENGTH_LONG).show();

                        // 도어락 열어주기
                        Wpapi.SetDoorLock_Open();
                        break;
                    }
                }

                // 등록 버튼 -> FLAG : 1
                if (FLAG == 1) {
                    button = findViewById(R.id.button);
                    button.setOnClickListener(new View.OnClickListener() {
                        public void onClick(View v) {

                            AlertDialog.Builder builder = new AlertDialog.Builder(FdActivity2.this);
                            builder.setTitle("카메라 앞에 서주세요.");
                            builder.setPositiveButton("확인", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialogInterface, int i) {
                                    // outputTensor -> 파일에 저장
                                    final float[] scores = outputTensor.getDataAsFloatArray();
                                    byte[] bb = floatArrayToBytes(scores);
                                    try {
                                        FileOutputStream fos = new FileOutputStream("/mnt/sdcard/face/" + userName + ".pt");
                                        fos.write(bb);
                                        fos.close();
                                        Toast.makeText(getApplicationContext(), "저장을 완료했습니다.", Toast.LENGTH_LONG).show();

                                        //Imgcodecs.imwrite("/mnt/sdcard/face/" + userName + ".jpg" , croppedImage);
                                    } catch (Exception e) {
                                        Log.e(TAG, e.getMessage());
                                    }
                                    FLAG = 0;
                                }
                            });
                            builder.show();
                        }
                    });
                }

            }
        }
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    // util function
    //

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
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

    public static float[] byteArrayToFloatArray(byte[] bytes) {
        float[] floats = new float[bytes.length / 4];
        for (int i = 0; i < floats.length; ++i)
            floats[i] = readFloatFromBytes(bytes, i * 4);
        return floats;
    }

    public static float readFloatFromBytes(byte[] bytes, int start) {
        Integer i = (bytes[start] << 24) & 0xff000000;
        i |= (bytes[start + 1] << 16) & 0x00ff0000 ;
        i |= (bytes[start + 2] << 8 ) & 0x0000ff00 ;
        i |= (bytes[start + 3]) & 0x000000ff;
        return Float.intBitsToFloat(i);
    }

    public static byte[] floatArrayToBytes(float[] d) {
        byte[] r = new byte[d.length * 4];
        for (int i = 0; i < d.length; i++) {
            byte[] s = floatToBytes(d[i]);
            for (int j = 0; j < 4; j++)
                r[4 * i + j] = s[j];
        }
        return r;
    }

    public static byte[] floatToBytes(float d) {
        int i = Float.floatToRawIntBits(d);
        return new byte[] {
                (byte) (i >> 24),
                (byte) (i >> 16),
                (byte) (i >> 8),
                (byte) (i) };
    }

    public static double cosine_similarity(float[] vec1, float[] vec2) {
        double cosim = vector_dot(vec1, vec2) / (vector_norm(vec1) * vector_norm(vec2));
        return cosim;
    }

    public static double vector_dot(float[] vec1, float[] vec2) {
        double sum = 0;
        for (int i = 0; i < vec1.length && i < vec2.length; i++)
            sum += vec1[i] * vec2[i];
        return sum;
    }

    public static double vector_norm(float[] vec) {
        double sum = 0;
        for (double v : vec)
            sum += v * v;
        return Math.sqrt(sum);
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
}
