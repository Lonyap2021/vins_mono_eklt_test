#include "feature_tracker.h"

int FeatureTracker::n_id = 0;
//  #define KLT 0
// int KLT = 1;

extern int KLT;


ofstream out("/home/ply/vins-mono/src/test.txt");
int keyF = 0;

std::vector<cv::Point2f> keyPoints;
int step = 1;

//判断pt是否在图像内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//根据status状态,将点进行重组，只保留跟踪到的点且在图像之内
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

//对跟踪点进行排序并依次选点，去除密集点使特征点分布均匀
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));//纯白色
    
    // prefer to keep features that are tracked for long time
    //保存长时间跟踪到的特征点
    //vector<pair<某一点跟踪次数，pair<某一点，某一点的id>>> cnt_pts_id
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    //操作对象是forw_pts,通过光流法在当前帧中仍然能追踪到的那些特征点在当前帧中的像素坐标！所以forw_pts里面放着的都是老特征点
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

   //对给定区间的所有元素进行排序，按照点的跟踪次数，从多到少进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            //降序排列
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255) //检测新建的mask在该点是否为255
        {
            //将跟踪到的点按照跟踪次数重新排列，并返回到forw_pts，ids，track_cnt
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //图片，点，半径，颜色为0表示在角点检测在该点不起作用,粗细（-1）表示填充
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

int image_width = COL;
int image_height = ROW;
void FeatureTracker::addPointsDense()
{
    int num = image_width/step*image_height/step - forw_pts.size();
    for(int i = 0;i<num;i++)
        ids.push_back(-1);
}

//添加新检测到的特征点n_pts，ID初始化-1，跟踪次数1
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}
int prev_klt_status = 0;
int prev_test = 0;
int size_dense = 0;


void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    KLT_STATUS=1;
    keyF++;
    if(KLT_STATUS){
        ROS_INFO("KLT_STATUS:%d",KLT_STATUS);
        readImageKlt(_img,_cur_time);
    }else{
        // readImageDense(_img,_cur_time);
        ROS_INFO("KLT_STATUS:%d",KLT_STATUS);
        readImageDense_test(_img,_cur_time); 
    }
}

void FeatureTracker::readImageKlt(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

   //若控制参数 EQUALIZE 为真,则调用 cv::creatCLAHE()对输入图像做自适应直方图均衡
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
        
    }
    //两个作用：1.在光流检测中存储跟踪到的点,2.在goodFeaturetotrack里存储检测到的角点
    //为了不使两者之间数据混乱，每次在进行1,2步时都要进行清空，goodFeaturetotrack里的forw_pts是在mask里清空的
    forw_pts.clear();
    
    //只有上一枕检测到角点，才能在这一帧进行光流跟踪
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
        //cur_pts:光流法需要找到的二维点的vector
        //forw_pts:输入特征在第二幅图像中计算出的新位置的二维点（单精度浮点坐标）的输出vector
        //status:如果对应特征的光流被发现，数组中的每一个元素都被设置为 1， 否则设置为 0。
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组
        reduceVector(ids, status);  
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++; //数值代表被追踪的次数，数值越大，说明被追踪的就越久

    if (PUB_THIS_FRAME)
    {
        rejectWithF();//通过基本矩阵剔除outliers
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//对跟踪点进行排序并去除密集点
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //光流跟踪新增加的检测角点个数
        //一共需要MAX_CNT个特征点，当前有static_cast(forw_pts.size())个特征点，
        //所以需要补充n_max_cnt = MAX_CNT - static_cast(forw_pts.size())个特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        //n_max_cnt=0,所明不需要跟踪
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //n_pts：存放检测的角点
            //MAX_CNT - forw_pts.size()：将检测的特征点数目
            //MIN_DIST：区分相邻两个角点的最小距离（小于这个距离得点将进行合并）
            //mask：它的维度必须和输入图像一致，且在mask值为0处不进行角点检测，目的是将以跟踪的点不进行角点检测，在其余
            //部分均匀的进行角点检测
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        //加入新增加的点
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    } 

    // if (cur_pts.size() > 0)
    //     getRT();
   
    if(cur_pts.size()<30){
        EKLT_FLAG = true;
        // PUB_THIS_FRAME = false;
    }else{
        EKLT_FLAG = false;
        // PUB_THIS_FRAME = true;
    }
    
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    prev_klt_status=KLT_STATUS;
   
}

int isnotFirstFram = 0;
void FeatureTracker::readImageDense(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
   
   //若控制参数 EQUALIZE 为真,则调用 cv::creatCLAHE()对输入图像做自适应直方图均衡
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if(isnotFirstFram==0){

        forw_img.release();
        
    }
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
        
        ids.clear();
        int num = 0;
        for(int y=0;y<_img.rows;y+=step)
            for(int x = 0;x<_img.cols;x+=step)
            {
                ids.push_back(num+size_dense*isnotFirstFram);
                num = num+1;
            }
        ROS_INFO("ids init size %d",ids.size());
        size_dense = num;
        isnotFirstFram++;

    }
    //两个作用：1.在光流检测中存储跟踪到的点,2.在goodFeaturetotrack里存储检测到的角点
    //为了不使两者之间数据混乱，每次在进行1,2步时都要进行清空，goodFeaturetotrack里的forw_pts是在mask里清空的
    forw_pts.clear();
    track_cnt.clear();
    cur_un_pts.clear();
    cur_pts.clear();

    //只有上一枕检测到角点，才能在这一帧进行光流跟踪
    // ids.clear();
    if(isnotFirstFram){
        
        cv::calcOpticalFlowFarneback(cur_img, forw_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cv::cvtColor(cur_img, cflow, cv::COLOR_GRAY2BGR);
        cv::Scalar color = cv::Scalar(0, 255, 0);
        vector<uchar> status;
        status.clear();
        
        for(int y=0;y<cflow.rows;y+=step)
            for(int x = 0;x<cflow.cols;x+=step)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cur_pts.push_back(cv::Point(x, y));
                forw_pts.push_back(cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)));
                if(fabs(fxy.x-0.0)<0.001 ||fabs(fxy.y-0.0)<0.001){
                    status.push_back(0);
                    track_cnt.push_back(1);
                    continue;   
                }
                status.push_back(1);
                track_cnt.push_back(2);
                // cv::line(cflow, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
                // cv::circle(cflow, cv::Point(x, y), 2, color, -1);
            } 
        ROS_DEBUG("cur_pts flow size is %d\n",cur_pts.size());
        // ROS_DEBUG("forw_pts flow size is %d\n",forw_pts.size());
        // ROS_INFO("status size is %d",status.size());
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组
        reduceVector(ids, status);
        // ROS_INFO("ids size is %d",ids.size());
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
    }

    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++;
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        
        //光流跟踪新增加的检测角点个数
        // int n_max_cnt = size_dense - static_cast<int>(forw_pts.size());  
        // if (n_max_cnt > 0)
        // {
        //     addPointsDense();
        // }
    }
    
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    prev_klt_status=KLT_STATUS;   
}

int restart_frame = 0;
void FeatureTracker::readImageDense_test(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    cv::Mat image = _img;

    forw_pts.clear();
    track_cnt.clear();
    cur_un_pts.clear();
    cur_pts.clear();
    prev_pts.clear();
    
   //若控制参数 EQUALIZE 为真,则调用 cv::creatCLAHE()对输入图像做自适应直方图均衡
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if(keyPoints.size()==0) {
        //将klt特征点给稠密进行追踪
        // for(int i = 0;i<cur_pts.size();i++){
        //     keyPoints.push_back(cur_pts[i]);
        // }
        //均匀取图片上的像素点进行追踪
        
        ids.clear();
        int num = 0;
        int start_row = ROW/2;
        // int start_row = 0;
        for(int y=start_row;y<_img.rows;y+=step)
            for(int x = 0;x<_img.cols;x+=step)
            {
                keyPoints.push_back(cv::Point(x, y)); 
                // ids.push_back(num+size_dense*restart_frame); 
                ids.push_back(num); 
                num++;      
            }
        
        ROS_DEBUG("keyPoints:%d",keyPoints.size());
        size_dense = num;
        forw_img.release();
        prev_img.release();
        cur_img.release();
    }

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        //稀疏光流选择的五个点，给到稠密光流，进行追踪
        // keyPoints.push_back(cv::Point(315.590668,356.079071));
        // keyPoints.push_back(cv::Point(210.187225,432.795380));
        // keyPoints.push_back(cv::Point(176.280899,402.470978));
        // keyPoints.push_back(cv::Point(379.619476,55.647018));
        // keyPoints.push_back(cv::Point(515.059937,200.531921)); 
    }
    else
    {
        forw_img = img;
        isnotFirstFram = 1;
    }
    //两个作用：1.在光流检测中存储跟踪到的点,2.在goodFeaturetotrack里存储检测到的角点
    //为了不使两者之间数据混乱，每次在进行1,2步时都要进行清空，goodFeaturetotrack里的forw_pts是在mask里清空的
    //只有上一枕检测到角点，才能在这一帧进行光流跟踪
    
    if(isnotFirstFram){
        isnotFirstFram++;
        cv::calcOpticalFlowFarneback(cur_img, forw_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cv::cvtColor(cur_img, cflow, cv::COLOR_GRAY2BGR);
        cv::Scalar color = cv::Scalar(0, 255, 0);
        vector<uchar> status;
        status.clear();
        
        for(int i =0;i<keyPoints.size();i++){
            int y = keyPoints[i].y;
            int x = keyPoints[i].x;
            
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cur_pts.push_back(cv::Point(x, y));
            forw_pts.push_back(cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)));
            if(fabs(fxy.x-0.0)<0.001 ||fabs(fxy.y-0.0)<0.001){
                    status.push_back(0);
                    track_cnt.push_back(1);
                    continue;   
            }
            status.push_back(1);
            track_cnt.push_back(2);
        } 
        // ROS_DEBUG("cur_pts flow size is %d\n",cur_pts.size());
        // ROS_DEBUG("forw_pts flow size is %d\n",forw_pts.size());
        // ROS_INFO("status size is %d",status.size());
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组
        
        reduceVector(ids, status);
        ROS_INFO("ids size is %d",ids.size());
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);  
    }

    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++;
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // setMask();
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // ROS_DEBUG("detect feature begins");
        // TicToc t_t;
        
        // //光流跟踪新增加的检测角点个数
        // int n_max_cnt = size_dense - static_cast<int>(forw_pts.size());  
        // if (n_max_cnt > 0)
        // {
        //     addPointsDense();
        // }
    }

    // if (isnotFirstFram)
    //     // getRT();
    
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    prev_klt_status=KLT_STATUS;

    if(isnotFirstFram>1){
        keyPoints.clear();
        keyPoints = forw_pts;
        if(ids.size()<30){
            keyPoints.clear();
            isnotFirstFram = 0;
            restart_frame++;
        }
    }

    
}

void FeatureTracker::readImageDenseKeyPoint_test(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
   
   //若控制参数 EQUALIZE 为真,则调用 cv::creatCLAHE()对输入图像做自适应直方图均衡
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if(keyPoints.size()==0) {
        for(int i = 0;i<cur_pts.size();i++){
            keyPoints.push_back(cur_pts[i]);
        }
    }

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        //稀疏光流选择的五个点，给到稠密光流，进行追踪
        // keyPoints.push_back(cv::Point(315.590668,356.079071));
        // keyPoints.push_back(cv::Point(210.187225,432.795380));
        // keyPoints.push_back(cv::Point(176.280899,402.470978));
        // keyPoints.push_back(cv::Point(379.619476,55.647018));
        // keyPoints.push_back(cv::Point(515.059937,200.531921)); 
    }
    else
    {
        forw_img = img;
        isnotFirstFram = 1;
    }
    //两个作用：1.在光流检测中存储跟踪到的点,2.在goodFeaturetotrack里存储检测到的角点
    //为了不使两者之间数据混乱，每次在进行1,2步时都要进行清空，goodFeaturetotrack里的forw_pts是在mask里清空的
    forw_pts.clear();
    track_cnt.clear();
    cur_un_pts.clear();
    cur_pts.clear();
    
    if(isnotFirstFram){
        isnotFirstFram++;
        cv::calcOpticalFlowFarneback(cur_img, forw_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cv::cvtColor(cur_img, cflow, cv::COLOR_GRAY2BGR);
        int step = 1;
        cv::Scalar color = cv::Scalar(0, 255, 0);
        vector<uchar> status;
        status.clear();
    
        for(int i =0;i<keyPoints.size();i++){
            int y = keyPoints[i].y;
            int x = keyPoints[i].x;
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cur_pts.push_back(cv::Point(x, y));
            forw_pts.push_back(cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)));
            status.push_back(1);
            track_cnt.push_back(2);
        } 
        
        ROS_DEBUG("cur_pts flow size is %d\n",cur_pts.size());
        // ROS_DEBUG("forw_pts flow size is %d\n",forw_pts.size());
        // ROS_INFO("status size is %d",status.size());
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组
        reduceVector(ids, status);
        // ROS_INFO("ids size is %d",ids.size());
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
    }

    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++;
    if (PUB_THIS_FRAME)
    {
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // setMask();
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // ROS_DEBUG("detect feature begins");
        // TicToc t_t;
        
        // //光流跟踪新增加的检测角点个数
        // int n_max_cnt = size_dense - static_cast<int>(forw_pts.size());  
        // if (n_max_cnt > 0)
        // {
        //     addPointsDense();
        // }
    }
    
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    prev_klt_status=KLT_STATUS;
    if(isnotFirstFram>1){
        keyPoints.clear();
        keyPoints = forw_pts;
    }
    
}

Vector3d Pref={0,0,0};

bool FeatureTracker::getRT(){

    Matrix3d relative_R;//定义容器
    Vector3d relative_T;

    if(!relativePose(relative_R, relative_T)){
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    Vector3d Pcur = relative_R * Pref+relative_T;
    Pref = Pcur;
    Pst = Pcur;
    std::cout<<"keyF"<<keyF<<endl;
    cout << relative_R<<endl;
    cout << relative_T<<endl;

    if(out.is_open()){
        out<<keyF<<"    "<<Pcur[0]<<"    "<<Pcur[1]<<"    "<<Pcur[2]<<endl;
    }
    return true;

}

bool FeatureTracker::relativePose(Matrix3d &relative_R, Vector3d &relative_T) //output array R,t
{
    if (ids.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(ids.size()); i++)
        {
            ll.push_back(cv::Point2f(cur_pts[i].x, cur_pts[i].y));
            rr.push_back(cv::Point2f(forw_pts[i].x, forw_pts[i].y));
        }
        cv::Mat mask;
        //因为这里的ll,rr是归一化坐标，所以得到的是本质矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        relative_R = R.transpose();
        relative_T = -R.transpose() * T;
        return true;
    }
    return false;

    
}


//对图像使用光流法进行特征点跟踪
void FeatureTracker::readImage1(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // KLT_STATUS = 0;
    // if(prev_klt_status!=KLT_STATUS){
    //     forw_pts.clear();
    //     cur_pts.clear();
    //     prev_img = cur_img;
    //     prev_pts.clear();
    //     prev_un_pts.clear(); 
    //     cur_un_pts.clear();   
    // }
   //若控制参数 EQUALIZE 为真,则调用 cv::creatCLAHE()对输入图像做自适应直方图均衡
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
        
    }
    //两个作用：1.在光流检测中存储跟踪到的点,2.在goodFeaturetotrack里存储检测到的角点
    //为了不使两者之间数据混乱，每次在进行1,2步时都要进行清空，goodFeaturetotrack里的forw_pts是在mask里清空的
    forw_pts.clear();
    KLT_STATUS = 0;
    // #if KLT
    if(KLT_STATUS){  
    //只有上一枕检测到角点，才能在这一帧进行光流跟踪
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
        //cur_pts:光流法需要找到的二维点的vector
        //forw_pts:输入特征在第二幅图像中计算出的新位置的二维点（单精度浮点坐标）的输出vector
        //status:如果对应特征的光流被发现，数组中的每一个元素都被设置为 1， 否则设置为 0。
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组

        reduceVector(ids, status);  

        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++; //数值代表被追踪的次数，数值越大，说明被追踪的就越久

    if (PUB_THIS_FRAME)
    {
        rejectWithF();//通过基本矩阵剔除outliers
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//对跟踪点进行排序并去除密集点
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //光流跟踪新增加的检测角点个数
        //一共需要MAX_CNT个特征点，当前有static_cast(forw_pts.size())个特征点，
        //所以需要补充n_max_cnt = MAX_CNT - static_cast(forw_pts.size())个特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        //n_max_cnt=0,所明不需要跟踪
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //n_pts：存放检测的角点
            //MAX_CNT - forw_pts.size()：将检测的特征点数目
            //MIN_DIST：区分相邻两个角点的最小距离（小于这个距离得点将进行合并）
            //mask：它的维度必须和输入图像一致，且在mask值为0处不进行角点检测，目的是将以跟踪的点不进行角点检测，在其余
            //部分均匀的进行角点检测
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        //加入新增加的点
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    }
    // #else
    else{
    
    //判断是否第二帧，开始进行光流跟踪
    // ids.clear();
    track_cnt.clear();
    cur_un_pts.clear();
    cur_pts.clear();

    if(cur_img.data){
        
        cv::calcOpticalFlowFarneback(cur_img, forw_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cv::cvtColor(cur_img, cflow, cv::COLOR_GRAY2BGR);
        int step = 18;
        cv::Scalar color = cv::Scalar(0, 255, 0);
        vector<uchar> status;
        status.clear();
        
        for(int y=0;y<cflow.rows;y+=step)
            for(int x = 0;x<cflow.cols;x+=step)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                // if(fabs(fxy.x-0.0)<0.001 ||fabs(fxy.y-0.0)<0.001){
                //     status.push_back(0);
                //     continue;   
                // }
                cur_pts.push_back(cv::Point(x, y));
                forw_pts.push_back(cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)));
                status.push_back(1);
                track_cnt.push_back(2);
                cv::line(cflow, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
                cv::circle(cflow, cv::Point(x, y), 2, color, -1);
            } 
        ROS_DEBUG("cur_pts flow size is %d\n",cur_pts.size());
        // ROS_DEBUG("forw_pts flow size is %d\n",forw_pts.size());
        // ROS_INFO("status size is %d",status.size());
        //判断跟踪的光流点，是否在图像内，跟踪的特征点在图像外状态数组内设为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组
        reduceVector(ids, status);
        // ROS_INFO("ids size is %d",ids.size());
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
    }

    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++;
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        
        //光流跟踪新增加的检测角点个数
        int n_max_cnt = size_dense - static_cast<int>(forw_pts.size());  
        if (n_max_cnt > 0)
        {
            addPointsDense();
        }
        /*
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        //n_max_cnt=0,所明不需要跟踪
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //n_pts：存放检测的角点
            //MAX_CNT - forw_pts.size()：将检测的特征点数目
            //MIN_DIST：区分相邻两个角点的最小距离（小于这个距离得点将进行合并）
            //mask：它的维度必须和输入图像一致，且在mask值为0处不进行角点检测，目的是将以跟踪的点不进行角点检测，在其余
            //部分均匀的进行角点检测
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        //加入新增加的点
        addPoints();
        
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        */
    }
    }
    // #endif
    
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    prev_klt_status=KLT_STATUS;
    
    if(!KLT_STATUS){
        if(prev_test==0){
            ids.clear();
            prev_test=1;
            int num = 0;
            int step = 18;
            for(int y=0;y<_img.rows;y+=step)
                for(int x = 0;x<_img.cols;x+=step)
                {
                    ids.push_back(num);
                    num = num+1;
                }
            ROS_INFO("ids init size %d",ids.size());
            size_dense = num;
        }
    }
 
}


//通过F矩阵去除outliers
void FeatureTracker::rejectWithF()
{
    //初始第一帧时不进行处理
    if(KLT_STATUS)
    {
        if (forw_pts.size() >= 8)
        {
            ROS_DEBUG("FM ransac begins");
            TicToc t_f;
            //分别是上一帧和当前帧去畸变的像素坐标
            vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
            for (unsigned int i = 0; i < cur_pts.size(); i++)
            {
                Eigen::Vector3d tmp_p;
                m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

                m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
            }
            vector<uchar> status;
                
             //status:在计算过程中没有被舍弃的点，元素被被置为1；否则置为0。
            cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
                
            int size_a = cur_pts.size();
            reduceVector(prev_pts, status);
            reduceVector(cur_pts, status);
            reduceVector(forw_pts, status);
            reduceVector(cur_un_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
            ROS_DEBUG("FM ransac costs: %fms", t_f.toc()); 
        }
    }
    else{
        // if (cur_img.data)
        if(isnotFirstFram)
        {
            ROS_DEBUG("FM ransac begins");
            TicToc t_f;
            //分别是上一帧和当前帧去畸变的像素坐标
            vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
            for (unsigned int i = 0; i < cur_pts.size(); i++)
            {
                Eigen::Vector3d tmp_p;
                m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

                m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
            }
            vector<uchar> status;
             //status:在计算过程中没有被舍弃的点，元素被被置为1；否则置为0。
            cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
                
            int size_a = cur_pts.size();
            reduceVector(prev_pts, status);
            reduceVector(cur_pts, status);
            reduceVector(forw_pts, status);
            reduceVector(cur_un_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
            ROS_DEBUG("FM ransac costs: %fms", t_f.toc()); 
        }        
    }
}
//更新特征点id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1){
            ids[i] = n_id++;
            
        }
        return true;
    }
    else
        return false;
}

//读取相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}
//显示去畸变矫正后的特征点
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
//对特征点的图像坐标去畸变矫正，并计算每个角点的速度
void FeatureTracker::undistortedPoints()
{
    
    cur_un_pts.clear();
    // cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        
        m_camera->liftProjective(a, b);

        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity

    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
