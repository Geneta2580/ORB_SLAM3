#include "ua_tag/TagOptimizer.h"

#include "Frame.h"
#include "CameraModels/GeometricCamera.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagMap.h"
#include "ua_tag/TagObservation.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#ifdef HAS_GTSAM
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#endif

namespace ORB_SLAM3 {
namespace tag {

namespace {

#ifdef HAS_GTSAM

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;
using ProjectionFactor =
    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

constexpr double kPixelSigma = 1.0;
constexpr double kChi2Mono = 5.991;  // 2 DoF, 95%
constexpr double kHuberDelta = 2.44774674741;  // sqrt(5.991)
constexpr int kMinInlierCorners = 3;
constexpr int kMaxIterations = 10;

gtsam::Pose3 ToGtsamPose(const Sophus::SE3f &T)
{
    return gtsam::Pose3(gtsam::Rot3(T.rotationMatrix().cast<double>()),
                        gtsam::Point3(T.translation().cast<double>()));
}

Sophus::SE3f FromGtsamPose(const gtsam::Pose3 &T)
{
    return Sophus::SE3f(T.rotation().matrix().cast<float>(),
                        T.translation().cast<float>());
}

boost::shared_ptr<gtsam::Cal3_S2> MakeCalib(GeometricCamera *camera,
                                            const Frame &frame)
{
    // 观测为去畸变角点，故使用理想针孔 Cal3_S2
    if(camera)
    {
        const Eigen::Matrix3f K = camera->toK_();
        return boost::make_shared<gtsam::Cal3_S2>(
            static_cast<double>(K(0, 0)), static_cast<double>(K(1, 1)), 0.0,
            static_cast<double>(K(0, 2)), static_cast<double>(K(1, 2)));
    }

    return boost::make_shared<gtsam::Cal3_S2>(
        static_cast<double>(frame.fx), static_cast<double>(frame.fy), 0.0,
        static_cast<double>(frame.cx), static_cast<double>(frame.cy));
}

struct CornerMeas
{
    gtsam::Point3 pw;
    gtsam::Point2 z;  // 去畸变后的像素坐标
    boost::shared_ptr<gtsam::Cal3_S2> K;
    boost::optional<gtsam::Pose3> body_P_sensor;
    TagObservation *obs = nullptr;
};

void CollectFixedTagCornerMeasurements(Frame &frame, const TagMap &tag_map,
                                       std::vector<CornerMeas> &meas)
{
    meas.clear();

    const auto K_left = MakeCalib(frame.mpCamera, frame);
    const auto K_right =
        MakeCalib(frame.mpCamera2 ? frame.mpCamera2 : frame.mpCamera, frame);

    boost::optional<gtsam::Pose3> T_body_right = boost::none;
    if(frame.mpCamera2)
        T_body_right = ToGtsamPose(frame.GetRelativePoseTlr());

    auto append_obs = [&](TagObservation &obs,
                          const boost::shared_ptr<gtsam::Cal3_S2> &K,
                          const boost::optional<gtsam::Pose3> &body_P_sensor) {
        if(!K)
            return;

        const auto map_tag = tag_map.GetTag(obs.tag_id);
        if(!map_tag || !map_tag->IsFixed() || !map_tag->HasWorldCorners())
            return;

        obs.is_outlier = false;
        const auto &pw = map_tag->GetWorldCorners();
        // 使用去畸变角点 + Cal3_S2 理想针孔投影
        const auto &uv = obs.corners_undistorted;

        for(int i = 0; i < 4; ++i)
        {
            CornerMeas m;
            m.pw = gtsam::Point3(pw[i].x(), pw[i].y(), pw[i].z());
            m.z = gtsam::Point2(uv[i].x, uv[i].y);
            m.K = K;
            m.body_P_sensor = body_P_sensor;
            m.obs = &obs;
            meas.push_back(m);
        }
    };

    for(TagObservation &obs : frame.mTagFrameData.left)
        append_obs(obs, K_left, boost::none);
    for(TagObservation &obs : frame.mTagFrameData.right)
        append_obs(obs, K_right, T_body_right);
}

gtsam::SharedNoiseModel MakeHuberNoise()
{
    const auto gaussian = gtsam::noiseModel::Isotropic::Sigma(2, kPixelSigma);
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(kHuberDelta), gaussian);
}

gtsam::SharedNoiseModel MakeGaussianNoise()
{
    return gtsam::noiseModel::Isotropic::Sigma(2, kPixelSigma);
}

void BuildGraphAndValues(const std::vector<CornerMeas> &meas,
                         const std::vector<char> &use_mask,
                         const gtsam::SharedNoiseModel &proj_noise,
                         const gtsam::Pose3 &Twc0,
                         gtsam::NonlinearFactorGraph &graph,
                         gtsam::Values &values)
{
    graph = gtsam::NonlinearFactorGraph();
    values.clear();
    values.insert(X(0), Twc0);

    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(!use_mask.empty() && !use_mask[i])
            continue;

        const CornerMeas &m = meas[i];
        const gtsam::Key point_key = L(static_cast<int>(i));

        // 3D 角点作为状态，用 NonlinearEquality 先验钉死（motion-only）
        values.insert(point_key, m.pw);
        graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Point3>>(point_key,
                                                                      m.pw);

        graph.emplace_shared<ProjectionFactor>(
            m.z, proj_noise, X(0), point_key, m.K, m.body_P_sensor);
    }
}

double CornerChi2(const CornerMeas &m, std::size_t idx, const gtsam::Pose3 &Twc)
{
    gtsam::Values values;
    values.insert(X(0), Twc);
    values.insert(L(static_cast<int>(idx)), m.pw);

    const ProjectionFactor factor(m.z, MakeGaussianNoise(), X(0),
                                  L(static_cast<int>(idx)), m.K,
                                  m.body_P_sensor);
    const gtsam::Vector e = factor.unwhitenedError(values);
    return e.squaredNorm() / (kPixelSigma * kPixelSigma);
}

#endif  // HAS_GTSAM

}  // namespace

void TagOptimizer::PoseOptimization(Frame &frame,
                                    const TagMap &tag_map,
                                    const std::optional<Sophus::SE3f> &Tcw_pred,
                                    bool verbose)
{
#ifndef HAS_GTSAM
    (void)tag_map;
    (void)Tcw_pred;
    if(verbose)
        std::cout << "[TagOptimizer] fail step=no_gtsam frame_id=" << frame.mnId
                  << std::endl;
    return;
#else
    Sophus::SE3f Tcw0;
    if(Tcw_pred)
        Tcw0 = *Tcw_pred;
    else if(frame.HasTagPose())
        Tcw0 = frame.GetTagPose();
    else
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=no_initial_pose frame_id="
                      << frame.mnId << std::endl;
        return;
    }

    // 收集固定 Tag 世界系角点的3D以及2D坐标作为测量
    std::vector<CornerMeas> meas;
    CollectFixedTagCornerMeasurements(frame, tag_map, meas);
    if(static_cast<int>(meas.size()) < kMinInlierCorners)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=too_few_measurements frame_id="
                      << frame.mnId << " n_meas=" << meas.size()
                      << " min_meas=" << kMinInlierCorners << std::endl;
        return;
    }

    // 进入优化的共视 Tag 数（每个 Tag 对应 4 个角点测量）
    std::unordered_set<int> covis_tag_ids;
    for(const CornerMeas &m : meas)
    {
        if(m.obs)
            covis_tag_ids.insert(m.obs->tag_id);
    }
    const int n_covis_tags = static_cast<int>(covis_tag_ids.size());
    if(n_covis_tags <= 0)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=no_covis_tags frame_id="
                      << frame.mnId << " n_meas=" << meas.size() << std::endl;
        return;
    }
    // 写入 pose 的门槛：内点角点数 > 3 * 共视 Tag 数
    const int min_inliers_to_accept = 3 * n_covis_tags;

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(kMaxIterations);
    lm_params.setVerbosityLM("SILENT");

    // 匀速运动假设提供初值
    const gtsam::Pose3 Twc_init = ToGtsamPose(Tcw0.inverse());  // Tcw -> Twc

    // 1) 全约束 + Huber；3D 点由 NonlinearEquality 固定
    std::vector<char> use_all(meas.size(), 1);
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    // 构建因子图和初始值（使用Huber鲁棒核，同时固定3D点）
    BuildGraphAndValues(meas, use_all, MakeHuberNoise(), Twc_init, graph, values);

    values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
    gtsam::Pose3 Twc = values.at<gtsam::Pose3>(X(0));

    // 2) chi2 标记外点（使用优化的Pose计算投影残差chi2）
    std::vector<char> inlier(meas.size(), 0);
    int n_inliers = 0;
    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(CornerChi2(meas[i], i, Twc) <= kChi2Mono)
        {
            inlier[i] = 1;
            ++n_inliers;
        }
    }

    if(n_inliers < kMinInlierCorners)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=huber_too_few_inliers frame_id="
                      << frame.mnId << " n_inliers=" << n_inliers
                      << " min_inliers=" << kMinInlierCorners
                      << " n_meas=" << meas.size()
                      << " n_covis_tags=" << n_covis_tags << std::endl;
        return;
    }

    // 3) 仅内点、去 Huber 再优化一轮
    BuildGraphAndValues(meas, inlier, MakeGaussianNoise(), Twc, graph, values);
    values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
    Twc = values.at<gtsam::Pose3>(X(0));

    // 最终再统计一次内点
    n_inliers = 0;
    for(std::size_t i = 0; i < meas.size(); ++i)
    {
        if(!inlier[i])
            continue;
        if(CornerChi2(meas[i], i, Twc) <= kChi2Mono)
            ++n_inliers;
        else
            inlier[i] = 0;
    }

    // 最终优化后回写观测级外点标记（≥3 个角点为外点则整条观测标外点）
    for(TagObservation &obs : frame.mTagFrameData.left)
        obs.is_outlier = false;
    for(TagObservation &obs : frame.mTagFrameData.right)
        obs.is_outlier = false;

    for(std::size_t i = 0; i < meas.size();)
    {
        TagObservation *obs = meas[i].obs;
        int n_bad = 0;
        std::size_t j = i;
        while(j < meas.size() && meas[j].obs == obs)
        {
            if(!inlier[j])
                ++n_bad;
            ++j;
        }
        if(obs && n_bad >= 3)
            obs->is_outlier = true;
        i = j;
    }

    // 仅当内点数 > 3 * 进入优化的共视 Tag 数时，用 BA Pose 覆盖 Frame.TagPose
    // 失败时保留调用方已写入的匀速预测兜底
    if(n_inliers <= min_inliers_to_accept)
    {
        if(verbose)
            std::cout << "[TagOptimizer] fail step=final_inlier_ratio frame_id="
                      << frame.mnId << " n_inliers=" << n_inliers
                      << " min_inliers_to_accept=" << min_inliers_to_accept
                      << " n_covis_tags=" << n_covis_tags
                      << " n_meas=" << meas.size() << std::endl;
        return;
    }

    frame.SetTagPose(FromGtsamPose(Twc).inverse());  // Twc -> Tcw
    if(verbose)
    {
        std::cout << "[TagOptimizer] frame_id=" << frame.mnId
                  << " n_inliers=" << n_inliers
                  << " n_covis_tags=" << n_covis_tags << std::endl;
    }
#endif
}

}  // namespace tag
}  // namespace ORB_SLAM3
