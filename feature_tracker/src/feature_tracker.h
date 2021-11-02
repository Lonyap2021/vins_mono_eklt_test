/*
 * @Descripttion: 
 * @version: 
 * @Author: Lonya Peng
 * @Date: 2021-10-15 15:41:21
 * @LastEditors: Lonya Peng
 * @LastEditTime: 2021-10-29 16:15:31
 */
#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

// #include "/home/ply/vins-mono/src/VINS-Mono/vins_estimator/src/estimator.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;



bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    //对新来的图像使用光流法进行特征点跟踪
    void readImage(const cv::Mat &_img,double _cur_time);

    void readImage1(const cv::Mat &_img,double _cur_time);
    
    void readImageKlt(const cv::Mat &_img,double _cur_time);

    void readImageDense(const cv::Mat &_img,double _cur_time);
    
    void readImageDense_test(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();
    
    void addPointsDense();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;//鱼眼相机mask，用来去除边缘噪点
    //cur_img和forw_img分别为光流跟踪的前后两帧
    //prev_img是上一次发布的帧，用处是，光流跟踪后用prev_img和forw_img根据rejectWithF()提出outlier
    cv::Mat prev_img, cur_img, forw_img,flow,cflow;
    
    vector<cv::Point2f> n_pts;//每一帧中新提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//对应的图像特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;//归一化相机坐标系下的坐标
    //当前帧相对前一帧特征点沿x,y方向的像素移动速度
    vector<cv::Point2f> pts_velocity;
    //能够被跟踪到的特征点的id
    vector<int> ids;
    vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数
    map<int, cv::Point2f> cur_un_pts_map; //构建id与归一化坐标的id，见undistortedPoints()
    map<int, cv::Point2f> prev_un_pts_map;
    //相机模型
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;
    static int n_id;//用来作为特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};
