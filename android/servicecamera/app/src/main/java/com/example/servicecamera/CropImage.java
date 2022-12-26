package com.example.servicecamera;
import static java.lang.Math.min;

import android.graphics.Bitmap;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class CropImage {

    public double[] _get_new_box(double src_w, double src_h, double[] bbox, double scale) {
        double x = bbox[0];
        double y = bbox[1];
        double box_w = bbox[2];
        double box_h = bbox[3];

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale));

        double new_width = box_w * scale;
        double new_height = box_h * scale;
        double center_x = box_w / 2 + x;
        double center_y = box_h / 2 + y;

        double left_top_x = center_x - new_width / 2;
        double left_top_y = center_y - new_height / 2;
        double right_bottom_x = center_x + new_width / 2;
        double right_bottom_y = center_y + new_height / 2;

        if (left_top_x< 0) {
            right_bottom_x -= left_top_x;
            left_top_x = 0;
        }

        if (left_top_y< 0){
            right_bottom_y -= left_top_y;
            left_top_y = 0;
        }

        if (right_bottom_x > src_w - 1) {
            left_top_x -= right_bottom_x - src_w + 1;
            right_bottom_x = src_w - 1;
        }

        if (right_bottom_y > src_h - 1) {
            left_top_y -= right_bottom_y - src_h + 1;
            right_bottom_y = src_h - 1;
        }
        double[] arr = new double[4];
        arr[0] = left_top_x;
        arr[1] = left_top_y;
        arr[2] = right_bottom_x;
        arr[3] = right_bottom_y;

        return arr;
    }

    Bitmap crop(Bitmap org_img, double[] bbox, double scale, int out_w, int out_h, boolean crop) {
        Bitmap dst_img = null;
        if(crop != true) {
            dst_img = Bitmap.createScaledBitmap(org_img, out_w, out_h, true);
        }
        else{
            double src_h = dst_img.getWidth();
            double src_w = dst_img.getHeight();
            double location[] = _get_new_box(src_w, src_h, bbox, scale);
            // left_top_x, left_top_y, right_bottom_x, right_bottom_y

            Bitmap img = cropBitmap(org_img, (int)(location[3] - location[1]+1), (int)(location[2]-location[0]+1));
            // org_img[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1];
            dst_img = Bitmap.createScaledBitmap(img, out_w, out_h, true);
        }
        return dst_img;
    }

    public Bitmap cropBitmap(Bitmap bitmap, int width, int height) {
        int originWidth = bitmap.getWidth();
        int originHeight = bitmap.getHeight();

        // 이미지를 crop 할 좌상단 좌표
        int x = 0;
        int y = 0;

        if (originWidth > width) { // 이미지의 가로가 view 의 가로보다 크면..
            x = (originWidth - width)/2;
        }

        if (originHeight > height) { // 이미지의 세로가 view 의 세로보다 크면..
            y = (originHeight - height)/2;
        }

        Bitmap cropedBitmap = Bitmap.createBitmap(bitmap, x, y, width, height);
        return cropedBitmap;
    }
}