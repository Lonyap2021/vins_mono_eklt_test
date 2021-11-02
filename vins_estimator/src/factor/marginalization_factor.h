#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);//添加参差块相关信息（优化变量，待marg的变量）
    void preMarginalize();//计算每个残差对应的雅克比，并更新parameter_block_data
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    int m, n;//m为要边缘化的变量个数，n为要保留下来的变量个数 m表示需marg变量/parameter_block_idx的总localSize n表示需保留变量的总localSize
    std::unordered_map<long, int> parameter_block_size; //global size//<优化变量内存地址,localSize>//为每个优化变量块的变量的大小，以IMU为例，是[7,9,7,9]
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size //<优化变量内存地址,在矩阵中的id>//被merge掉的变量
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,数据>//每个优化块数据
    //他们的key都同一是long类型的内存地址，而value分别是，各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据。
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;
    //他们是进行边缘化之后保留下来的各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据
    Eigen::MatrixXd linearized_jacobians;//分别指的是边缘化之后从信息矩阵恢复出来雅克比矩阵和残差向量
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
