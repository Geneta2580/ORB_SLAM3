#ifdef HAS_APRILTAG

#include "ua_tag/AprilTagDetector.h"

#include "Settings.h"
#include "CameraModels/GeometricCamera.h"

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include "apriltag.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ORB_SLAM3
{
namespace ua_tag
{

namespace
{

apriltag_family_t *CreateTagFamily(const std::string &family)
{
    if(family == "tag36h11")
        return tag36h11_create();
    if(family == "tag25h9")
        return tag25h9_create();
    if(family == "tag16h5")
        return tag16h5_create();
    if(family == "tagCircle21h7")
        return tagCircle21h7_create();
    if(family == "tagCircle49h12")
        return tagCircle49h12_create();
    if(family == "tagStandard41h12")
        return tagStandard41h12_create();
    if(family == "tagStandard52h13")
        return tagStandard52h13_create();
    if(family == "tagCustom48h12")
        return tagCustom48h12_create();
    return NULL;
}

void DestroyTagFamily(apriltag_family_t *tf, const std::string &family)
{
    if(tf == NULL)
        return;

    if(family == "tag36h11")
        tag36h11_destroy(tf);
    else if(family == "tag25h9")
        tag25h9_destroy(tf);
    else if(family == "tag16h5")
        tag16h5_destroy(tf);
    else if(family == "tagCircle21h7")
        tagCircle21h7_destroy(tf);
    else if(family == "tagCircle49h12")
        tagCircle49h12_destroy(tf);
    else if(family == "tagStandard41h12")
        tagStandard41h12_destroy(tf);
    else if(family == "tagStandard52h13")
        tagStandard52h13_destroy(tf);
    else if(family == "tagCustom48h12")
        tagCustom48h12_destroy(tf);
}

// 将图像转换为灰度图像
cv::Mat ToGray(const cv::Mat &image)
{
    if(image.empty())
        throw std::invalid_argument("AprilTagDetector: empty image");

    cv::Mat gray;
    if(image.channels() == 1)
        gray = image;
    else if(image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if(image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        throw std::invalid_argument("AprilTagDetector: unsupported channel count");

    if(!gray.isContinuous())
        gray = gray.clone();
    return gray;
}

Sophus::SE3f MakeTct(const cv::Matx33d &R, const cv::Vec3d &t)
{
    Eigen::Matrix3f Rf;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            Rf(r, c) = static_cast<float>(R(r, c));
    return Sophus::SE3f(Rf, Eigen::Vector3f(static_cast<float>(t[0]),
                                            static_cast<float>(t[1]),
                                            static_cast<float>(t[2])));
}

// 特征点去畸变
void UndistortCorners(const tag::TagObservation &src, const CameraModel &camera,
                      std::array<cv::Point2f, 4> &out)
{
    bool has_distortion = false;
    if(!camera.dist_coeffs.empty())
    {
        for(int i = 0; i < camera.dist_coeffs.total(); ++i)
        {
            if(std::abs(camera.dist_coeffs.at<double>(i)) > 1e-12)
            {
                has_distortion = true;
                break;
            }
        }
    }

    if(!has_distortion)
    {
        out = src.corners_raw;
        return;
    }

    std::vector<cv::Point2f> pts(src.corners_raw.begin(), src.corners_raw.end());
    std::vector<cv::Point2f> undist;
    cv::undistortPoints(pts, undist, cv::Mat(camera.K()), camera.dist_coeffs, cv::noArray(), cv::Mat(camera.K()));
    for(int k = 0; k < 4; ++k)
        out[k] = undist[k];
}

double ComputeReprojectionError(const std::vector<cv::Point3f> &object_pts,
                                const std::vector<cv::Point2f> &image_pts,
                                const cv::Mat &rvec,
                                const cv::Mat &tvec,
                                const CameraModel &camera)
{
    if(object_pts.empty() || object_pts.size() != image_pts.size())
        return std::numeric_limits<double>::infinity();

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_pts, rvec, tvec, cv::Mat(camera.K()), camera.dist_coeffs, projected);
    if(projected.size() != image_pts.size())
        return std::numeric_limits<double>::infinity();

    double sum_sq = 0.0;
    for(size_t i = 0; i < image_pts.size(); ++i)
    {
        const double dx = projected[i].x - image_pts[i].x;
        const double dy = projected[i].y - image_pts[i].y;
        sum_sq += dx * dx + dy * dy;
    }
    return std::sqrt(sum_sq / static_cast<double>(image_pts.size()));
}

// 相对预测的旋转角偏差（弧度）：R_err = R_pred^T * R_cand
double RotationDeltaToPrediction(const tag::TagPoseCandidate &cand, const PosePrediction &pred)
{
    const Eigen::Matrix3f Rf = cand.T_ct.rotationMatrix();
    cv::Matx33d R_cand;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            R_cand(r, c) = static_cast<double>(Rf(r, c));

    const cv::Matx33d R_err = pred.R.t() * R_cand;
    const double trace = R_err(0, 0) + R_err(1, 1) + R_err(2, 2);
    const double cos_angle = std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)));
    return std::acos(cos_angle);
}

// 相对预测的平移欧氏偏差（米）
double TranslationDeltaToPrediction(const tag::TagPoseCandidate &cand, const PosePrediction &pred)
{
    const Eigen::Vector3f tf = cand.T_ct.translation();
    const cv::Vec3d dt = cv::Vec3d(tf[0], tf[1], tf[2]) - pred.t;
    return std::sqrt(dt.dot(dt));
}

// 第 2/3 层歧义消解。
// 输入 candidates 必须已经通过：正深度检查（候选生成阶段）。
// 本函数内再做：数值有效性、绝对重投影 RMSE、误差比、时序门控/代价。
// IPPE_SQUARE 最多 2 个候选；num_candidates > 2 直接失败。
// 返回 selected_candidate；不可信 / 未消歧时返回 -1（正式返回值不做“临时最优”）。
// 若双解均有效，写入 ambiguity_ratio = e_worse / e_better。
int ResolvePoseAmbiguity(const std::array<tag::TagPoseCandidate, 2> &candidates,
                         int num_candidates,
                         const PosePrediction *prediction,
                         int tag_id,
                         float &ambiguity_ratio)
{
    ambiguity_ratio = -1.0f;

    if(num_candidates <= 0 || num_candidates > 2)
        return -1;

    constexpr double kMaxReprojectionRmse = 3.0;  // 需标定
    constexpr double kTauRhoStrong = 3.0;
    constexpr double kErrorEps = 1e-9; // 避免除零

    int i_better = 0;
    int i_worse = -1;
    if(num_candidates == 2)
    {
        i_worse = 1;
        if(candidates[1].reprojection_error < candidates[0].reprojection_error)
        {
            i_better = 1;
            i_worse = 0;
        }
    }

    const double e1 = static_cast<double>(candidates[i_better].reprojection_error);
    if(!std::isfinite(e1) || e1 > kMaxReprojectionRmse)
        return -1;

    // 只有一个有效候选
    if(num_candidates == 1)
        return i_better;

    const double e2 = static_cast<double>(candidates[i_worse].reprojection_error);
    if(!std::isfinite(e2))
        return -1;

    // ρ = e2/e1（加 eps 避免 e1=e2=0 被当成强支持）
    const double rho = (e2 + kErrorEps) / (e1 + kErrorEps);
    ambiguity_ratio = static_cast<float>(rho);
    const bool strong_reproj = (rho > kTauRhoStrong);

    const bool has_pred = prediction && prediction->valid &&
                          (prediction->tag_id < 0 || prediction->tag_id == tag_id);

    // -----------------------------------------------------
    // 强重投影证据：优先最小误差解；有预测时仅做宽松异常检查
    // -----------------------------------------------------
    if(strong_reproj)
    {
        if(has_pred)
        {
            constexpr double kLooseTauR = 1.0;   // rad，需标定
            constexpr double kLooseTauT = 0.30;  // m，需标定

            const double dtheta =
                RotationDeltaToPrediction(candidates[i_better], *prediction);
            const double dtrans =
                TranslationDeltaToPrediction(candidates[i_better], *prediction);

            // 图像与可靠预测严重冲突
            if(dtheta >= kLooseTauR || dtrans >= kLooseTauT)
                return -1;
        }

        return i_better;
    }

    // -----------------------------------------------------
    // 重投影无法消歧，又没有时序预测 → 不可信
    // -----------------------------------------------------
    if(!has_pred)
        return -1;

    // -----------------------------------------------------
    // 弱重投影证据：严格时序门控 → 门内综合代价（要求足够代价差）
    // -----------------------------------------------------
    constexpr double kTauR = 0.5;   // rad
    constexpr double kTauT = 0.15;  // m

    double dtheta[2];
    double dtrans[2];
    bool pass[2];
    int n_pass = 0;
    int sole_pass = -1;

    for(int i = 0; i < 2; ++i)
    {
        dtheta[i] = RotationDeltaToPrediction(candidates[i], *prediction);
        dtrans[i] = TranslationDeltaToPrediction(candidates[i], *prediction);
        pass[i] = (dtheta[i] < kTauR) && (dtrans[i] < kTauT);
        if(pass[i])
        {
            ++n_pass;
            sole_pass = i;
        }
    }

    if(n_pass == 0)
        return -1;

    if(n_pass == 1)
        return sole_pass;

    // 两个均通过时序门控，计算综合代价
    constexpr double kWe = 1.0;
    constexpr double kWR = 1.0;
    constexpr double kWt = 1.0;
    constexpr double kSigmaE = 1.0;   // px
    constexpr double kSigmaR = 0.2;   // rad
    constexpr double kSigmaT = 0.05;  // m
    constexpr double kMinJGap = 0.2;  // 需标定

    double J[2];
    for(int i = 0; i < 2; ++i)
    {
        J[i] = kWe * static_cast<double>(candidates[i].reprojection_error) / kSigmaE +
               kWR * dtheta[i] / kSigmaR +
               kWt * dtrans[i] / kSigmaT;
    }

    const int best = (J[0] <= J[1]) ? 0 : 1;
    const int second = 1 - best;

    // 综合代价过于接近，不声称已消歧
    if(J[second] - J[best] < kMinJGap)
        return -1;

    return best;
}

}  // namespace

struct AprilTagDetector::Impl
{
    apriltag_detector_t *detector;
    apriltag_family_t *family;
    std::string family_name;
    double tag_size;

    Impl(const AprilTagDetectorConfig &config)
        : detector(NULL), family(NULL), family_name(config.family), tag_size(config.tag_size)
    {
        family = CreateTagFamily(config.family);
        if(family == NULL)
            throw std::invalid_argument("AprilTagDetector: unsupported tag family: " + config.family);

        detector = apriltag_detector_create();
        apriltag_detector_add_family_bits(detector, family, config.hamming);

        detector->quad_decimate = config.quad_decimate;
        detector->quad_sigma = config.quad_sigma;
        detector->nthreads = config.nthreads;
        detector->refine_edges = config.refine_edges;
    }

    ~Impl()
    {
        if(detector != NULL)
        {
            apriltag_detector_destroy(detector);
            detector = NULL;
        }
        DestroyTagFamily(family, family_name);
        family = NULL;
    }
};

// 复用ORB-SLAM3的Settings，获取相机模型
CameraModel CameraModel::FromSettings(Settings *settings)
{
    if(!settings)
        throw std::invalid_argument("CameraModel::FromSettings: settings is null");

    if(settings->cameraType() == Settings::KannalaBrandt)
        throw std::runtime_error("CameraModel::FromSettings: KannalaBrandt is not supported for AprilTag PnP");

    GeometricCamera *geom = settings->camera1();
    if(!geom)
        throw std::runtime_error("CameraModel::FromSettings: camera1 is null");

    CameraModel cam;
    cam.fx = static_cast<double>(geom->getParameter(0));
    cam.fy = static_cast<double>(geom->getParameter(1));
    cam.cx = static_cast<double>(geom->getParameter(2));
    cam.cy = static_cast<double>(geom->getParameter(3));

    cv::Mat dist = settings->camera1DistortionCoef();
    if(!dist.empty())
        dist.convertTo(cam.dist_coeffs, CV_64F);

    return cam;
}

// 从yaml文件中读取AprilTagDetector的配置
AprilTagDetectorConfig AprilTagDetectorConfig::FromYaml(const std::string &settingsFile)
{
    AprilTagDetectorConfig config;

    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("AprilTagDetectorConfig: failed to open settings file: " + settingsFile);

    cv::FileNode node;
    node = fs["Tag.family"];
    if(!node.empty() && node.isString())
        config.family = static_cast<std::string>(node);

    node = fs["Tag.hamming"];
    if(!node.empty())
        config.hamming = static_cast<int>(node);

    node = fs["Tag.quad_decimate"];
    if(!node.empty())
        config.quad_decimate = static_cast<double>(node);

    node = fs["Tag.quad_sigma"];
    if(!node.empty())
        config.quad_sigma = static_cast<double>(node);

    node = fs["Tag.nthreads"];
    if(!node.empty())
        config.nthreads = static_cast<int>(node);

    node = fs["Tag.refine_edges"];
    if(!node.empty())
        config.refine_edges = static_cast<int>(node) != 0;

    node = fs["Tag.size"];
    if(!node.empty())
        config.tag_size = static_cast<double>(node);

    return config;
}

// 默认初始化AprilTagDetector
AprilTagDetector::AprilTagDetector(const AprilTagDetectorConfig &config)
    : mpImpl(new Impl(config))
{
}

// 从yaml文件中初始化AprilTagDetector
AprilTagDetector::AprilTagDetector(const std::string &settingsFile)
    : mpImpl(new Impl(AprilTagDetectorConfig::FromYaml(settingsFile)))
{
}

// 销毁AprilTagDetector
AprilTagDetector::~AprilTagDetector()
{
    delete mpImpl;
}

// 检测AprilTags的角点
bool AprilTagDetector::DetectCorners(const cv::Mat &image, std::vector<tag::TagObservation> &observations)
{
    observations.clear();

    cv::Mat gray = ToGray(image);

    image_u8_t apriltag_image = {
        gray.cols,
        gray.rows,
        gray.cols,
        gray.data
    };

    zarray_t *raw = apriltag_detector_detect(mpImpl->detector, &apriltag_image);
    if(raw == NULL)
        return false;

    const int n = zarray_size(raw);
    observations.reserve(n);

    for(int i = 0; i < n; ++i)
    {
        apriltag_detection_t *det = NULL;
        zarray_get(raw, i, &det);
        if(det == NULL)
            continue;

        tag::TagObservation obs;
        obs.tag_id = det->id;
        obs.camera_id = tag::CameraId::LEFT_OR_MONO;
        obs.hamming = det->hamming;
        obs.decision_margin = det->decision_margin;
        for(int k = 0; k < 4; ++k)
        {
            obs.corners_raw[k] = cv::Point2f(static_cast<float>(det->p[k][0]),
                                             static_cast<float>(det->p[k][1]));
            // 去畸变前先拷贝；EstimatePose 时再写入真正的去畸变结果
            obs.corners_undistorted[k] = obs.corners_raw[k];
        }
        // pose_estimate 保持 nullopt：DetectCorners 只负责原始角点
        observations.push_back(obs);
    }

    apriltag_detections_destroy(raw);
    return true;
}

// 估计AprilTags的位姿
bool AprilTagDetector::EstimatePose(tag::TagObservation &observation,
                                    const CameraModel &camera,
                                    const PosePrediction *prediction)
{
    observation.pose_estimate = std::nullopt;

    if(mpImpl->tag_size <= 0.0 || camera.fx <= 0.0 || camera.fy <= 0.0)
        return false;

    // 去畸变角点写入观测；PnP 仍用原始角点 + 畸变系数
    UndistortCorners(observation, camera, observation.corners_undistorted);

    // 构建AprilTag的物体点，Tag自身坐标系下的4个角点3D坐标
    std::vector<cv::Point3f> object_pts;
    BuildSquareObjectPoints(mpImpl->tag_size, object_pts);

    // PnP 使用原始角点 + 畸变系数（与 OpenCV 推荐一致）；无畸变时等价于理想针孔
    std::vector<cv::Point2f> image_pts(observation.corners_raw.begin(),
                                       observation.corners_raw.end());

    std::vector<cv::Mat> rvecs, tvecs;
    int nsol = cv::solvePnPGeneric(object_pts, image_pts,
                                   cv::Mat(camera.K()), camera.dist_coeffs,
                                   rvecs, tvecs,
                                   false, cv::SOLVEPNP_IPPE_SQUARE);
    if(nsol <= 0 || rvecs.empty() || tvecs.empty())
        return false;

    tag::TagPoseEstimate estimate;

    // 仅正深度解进入候选（只保留有效 IPPE 解）
    int num_candidates = 0;
    for(int i = 0; i < nsol && num_candidates < 2; ++i)
    {
        const cv::Vec3d t(tvecs[i].at<double>(0),
                          tvecs[i].at<double>(1),
                          tvecs[i].at<double>(2));
        if(t[2] <= 0.0)
            continue;

        cv::Matx33d R;
        cv::Rodrigues(rvecs[i], R);

        tag::TagPoseCandidate &cand = estimate.candidates[num_candidates++];
        cand.T_ct = MakeTct(R, t);
        cand.reprojection_error = static_cast<float>(ComputeReprojectionError(
            object_pts, image_pts, rvecs[i], tvecs[i], camera));
        cand.valid = true;
    }

    if(num_candidates <= 0)
        return false;

    // IPPE_SQUARE 期望 2 个有效解；不足则不写入 pose_estimate
    if(num_candidates < 2)
        return false;

    float ambiguity_ratio = -1.0f;
    const int best = ResolvePoseAmbiguity(estimate.candidates,
                                          num_candidates,
                                          prediction,
                                          observation.tag_id,
                                          ambiguity_ratio);
    if(best < 0)
    {
        // 已有候选但未消歧：仍写入 estimate，selected_candidate 保持 -1
        estimate.ambiguity_ratio = ambiguity_ratio;
        observation.pose_estimate = estimate;
        return false;
    }

    estimate.selected_candidate = best;
    estimate.ambiguity_ratio = ambiguity_ratio;
    observation.pose_estimate = estimate;
    return true;
}

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif  // HAS_APRILTAG
