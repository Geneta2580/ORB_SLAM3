#ifndef UA_TAG_APRILTAG_DETECTOR_H
#define UA_TAG_APRILTAG_DETECTOR_H

#include "ua_tag/TagObservation.h"

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace ORB_SLAM3
{

class Settings;

namespace ua_tag
{

// 相机针孔模型（用于 PnP）；畸变可为空表示已校正/无畸变
struct CameraModel
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    cv::Mat dist_coeffs;  // 空、4 或 5 维 (k1,k2,p1,p2[,k3])

    cv::Matx33d K() const
    {
        return cv::Matx33d(fx, 0.0, cx,
                           0.0, fy, cy,
                           0.0, 0.0, 1.0);
    }

    // 复用 ORB-SLAM3 已加载的 Settings（含可能的 newWidth/newHeight 缩放）
    static CameraModel FromSettings(Settings *settings);
};

// 可选时序先验：某 tag 的上一帧 T_ct，用于二义性消解
struct PosePrediction
{
    int tag_id = -1;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t = cv::Vec3d(0, 0, 0);
    bool valid = false;
};

struct AprilTagDetectorConfig
{
    std::string family = "tag36h11";
    int hamming = 1;
    double quad_decimate = 2.0;
    double quad_sigma = 0.0;
    int nthreads = 1;
    bool refine_edges = true;
    double tag_size = 0.16;  // Tag 边长（米），用于 IPPE_SQUARE

    // 从 ORB-SLAM3 settings yaml 读取 Tag.*（缺省保留默认值）
    static AprilTagDetectorConfig FromYaml(const std::string &settingsFile);
};

// AprilTag 角点顺序与 OpenCV IPPE_SQUARE 物体点一致：
// (-s/2,s/2,0), (s/2,s/2,0), (s/2,-s/2,0), (-s/2,-s/2,0)
inline void BuildSquareObjectPoints(double tag_size, std::vector<cv::Point3f> &object_pts)
{
    const float h = static_cast<float>(tag_size) * 0.5f;
    object_pts.clear();
    object_pts.reserve(4);
    object_pts.emplace_back(-h,  h, 0.f);
    object_pts.emplace_back( h,  h, 0.f);
    object_pts.emplace_back( h, -h, 0.f);
    object_pts.emplace_back(-h, -h, 0.f);
}

class AprilTagDetector
{
public:
    explicit AprilTagDetector(const AprilTagDetectorConfig &config = AprilTagDetectorConfig());
    explicit AprilTagDetector(const std::string &settingsFile);
    ~AprilTagDetector();

    // 禁止拷贝：Impl 持有 apriltag 裸指针资源，拷贝会导致双重释放
    AprilTagDetector(const AprilTagDetector &) = delete;
    AprilTagDetector &operator=(const AprilTagDetector &) = delete;

    // 仅角点检测：填充 tag_id / corners / hamming / decision_margin（不做 IPPE）
    bool DetectCorners(const cv::Mat &image, std::vector<tag::TagObservation> &observations);

    // IPPE_SQUARE PnP；prediction 可选，为空则不做时序消歧
    bool EstimatePose(tag::TagObservation &observation,
                      const CameraModel &camera,
                      const PosePrediction *prediction = nullptr);

private:
    struct Impl;
    Impl *mpImpl;
};

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif
