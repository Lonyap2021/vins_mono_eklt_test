#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

//三角化求深度
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	//a．第一次筛选：把滑窗的所有特征点中，那些没有3D坐标的点pass掉。
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)//feature_num = sfm_f.size() line121
	{//要把待求帧i上所有特征点的归一化坐标和3D坐标(l系上)都找出来
		if (sfm_f[j].state != true)//这个特征点没有被三角化为空间点，跳过这个点的PnP
			continue;
		//b．因为是对当前帧和上一帧进行PnP，所以这些有3D坐标的特征点，不仅得在当前帧被观测到，还得在上一帧被观测到。
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)//依次遍历特征j在每一帧中的归一化坐标
		{
			if (sfm_f[j].observation[k].first == i)//如果该特征在帧i上出现过
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);//把在待求帧i上出现过的特征的归一化坐标放到容器中
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);//把在待求帧i上出现过的特征在参考系l的空间坐标放到容器中
				break;//因为一个特征在帧i上只会出现一次，一旦找到了就没有必要再继续找了
			}
		}
	}
	//c.如果这些有3D坐标的特征点，并且在当前帧和上一帧都出现了，数量却少于15，那么整个初始化全部失败。因为它的是层层往上传递。
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	// d．套用openCV的公式，进行PnP求解
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);//转换成solvePnP能处理的格式
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);//得到了第i帧到第l帧的旋转平移
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);//转换成原有格式
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;//覆盖原先的旋转平移
	P_initial = T_pnp;
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)//在所有特征里面依次寻找
	{
		if (sfm_f[j].state == true)//如果这个特征已经三角化过了，那就跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)//如果这个特征在frame0出现过
			{
				point0 = sfm_f[j].observation[k].second;//把他的归一化坐标提取出来
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)//如果这个特征在frame1出现过
			{
				point1 = sfm_f[j].observation[k].second;//把他的归一化坐标提取出来
				has_1 = true;
			}
		}
		if (has_0 && has_1)//如果这两个归一化坐标都存在
		{
			Vector3d point_3d;
			//首先他把sfm_f的特征点取出来，一个个地检查看看这个特征点是不是被2帧都观测到了，如果被观测到了，再执行三角化操作
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);//根据他们的位姿和归一化坐标，输出在参考系l下的的空间坐标
			sfm_f[j].state = true;//已经完成三角化，状态更改为true
			sfm_f[j].position[0] = point_3d(0);//把参考系l下的的空间坐标赋值给这个特征点的对象
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)

bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	//(1) 把第l帧作为参考坐标系，获得最新一帧在参考坐标系下的位姿 
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	q[l].w() = 1;//参考帧的四元数，平移为1和0
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();//1、这里把第l帧看作参考坐标系，根据当前帧到第l帧的relative_R，relative_T，得到当前帧在参考坐标系下的位姿，之后的pose[i]表示第l帧到第i帧的变换矩阵[R|T]
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);//frame_num-1表示当前帧* relative c0_->ck
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;
	//(2) 构造容器，存储滑窗内 第l帧 相对于 其它帧 和 最新一帧 的位姿
	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];
	//注意，这些容器存储的都是相对运动，大写的容器对应的是l帧旋转到各个帧。 小写的容器是用于全局BA时使用的，也同样是l帧旋转到各个帧。
	//之所以在这两个地方要保存这种相反的旋转，是因为三角化求深度的时候需要这个相反旋转的矩阵！ 为了表示区别，称这两类容器叫 坐标系变换矩阵，而不能叫 位姿 ！ 
	
	//(3) 对于第l帧和最新一帧，它们的相对运动是已知的，可以直接放入容器   
	//从l帧旋转到各个帧的旋转平移
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];
	//这块有一个取相反旋转的操作，因为三角化的时候需要这个相反的旋转

	
	// (5)对于在sliding window里在第l帧之后的每一帧，分别都和前一帧用PnP求它的位姿，得到位姿后再和最新一帧三角化得到它们共视点的3D坐标   
	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	//求位姿，2D-2D对极几何只是在第一次使用，也就是没有3D特征点坐标的时候使用，一旦有了特征点，之后都会用3D-2D的方式求位姿。
	//然后会进入PnP求新位姿，然后三角化求新3D坐标的循环中
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			//已知第i帧上出现的一些特征点的l系上空间坐标，通过上一帧的旋转平移得到下一帧的旋转平移
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;//SfM失败
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// (4)三角化l帧和最新帧，获得他们的共视点在l帧上的空间坐标 注意，三角化的前提有1个：两帧的(相对)位姿已知。这样才能把他们的共视点的三维坐标还原出来。
	 // triangulateTwoFrames(l, Pose[l], frame_num - 1, Pose[frame_num - 1], sfm_f); 我们看一下这个函数的内容。
		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//(6)从第l+1帧到滑窗的最后的每一帧再与第l帧进行三角化补充3D坐标 现在回到construct()函数，在上一步，求出了l帧后面的每一帧的位姿，也求出了它们相对于最后一帧的共视点的3D坐标，
	//但是这是不够的，现在继续补充3D坐标，那么就和第l帧进行三角化。
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    // (7)对于在sliding window里在第l帧之前的每一帧，分别都和后一帧用PnP求它的位姿，得到位姿后再和第l帧三角化得到它们共视点的3D坐标   
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	// l帧之后的帧都有着落了，现在解决之前的帧。这个过程和(5)完全一样。
	// (8) 三角化其他未恢复的特征点 至此得到了滑动窗口中所有图像帧的位姿以及特征点的3D坐标。   
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)//里面存放着第j个特征在滑窗所有帧里的归一化坐标
		{
			Vector2d point0, point1;
  			int frame_0 = sfm_f[j].observation[0].first;//第j个特征在滑窗第一次被观测到的帧的ID
        	point0 = sfm_f[j].observation[0].second;//第j个特征在滑窗第一次被观测到的帧的归一化坐标
        	int frame_1 = sfm_f[j].observation.back().first;//第j个特征在滑窗最后一次被观测到的帧的ID
        	point1 = sfm_f[j].observation.back().second;//第j特征在滑窗最后一次被观测到的帧的归一化坐标

			Vector3d point_3d;//在帧l下的空间坐标
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
 	//(9)采用ceres进行全局BA a.声明problem 注意，因为四元数是四维的，但是自由度是3维的，因此需要引入LocalParameterization。
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	//b.加入待优化量：全局位姿   
	for (int i = 0; i < frame_num; i++)//7、使用cares进行全局BA优化
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		//  在这里，可以发现，仅仅是位姿被优化了，特征点的3D坐标没有被优化！ 
		//c.固定先验值 因为l帧是参考系，最新帧的平移也是先验，如果不固定住，原本可观的量会变的不可观。
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}
	//d.加入残差块 这里采用的仍然是最小化重投影误差的方式，所以需要2D-3D信息，注意这块没有加loss function
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}
	// e.shur消元求解   shur消元有2大作用，一个是在最小二乘中利用H矩阵稀疏的性质进行加速求解，
	//另一个是在sliding window时求解marg掉老帧后的先验信息矩阵。这块是shur消元的第一个用法。
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	 
	//d．返回特征点l系下3D坐标和优化后的全局位姿   
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;
	//优化完成后，需要获得各帧在帧l系下的位姿(也就是各帧到l帧的旋转平移)，所以需要inverse操作，然后把特征点在帧l系下的3D坐标传递出来。
	 
}

