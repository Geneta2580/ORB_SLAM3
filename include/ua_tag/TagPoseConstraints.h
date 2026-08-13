#ifndef UA_TAG_TAG_POSE_CONSTRAINTS_H
#define UA_TAG_TAG_POSE_CONSTRAINTS_H

#include "ua_tag/TagObservation.h"

#include <array>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Frame;
class KeyFrame;
class Map;
class MapPoint;

namespace tag {

// 只读快照：优化期间不持有 Map/MapTag 锁
struct TagPoseConstraint
{
    int tagId = -1;
    std::array<Eigen::Vector3d, 4> worldCorners{};
};

struct TagPoseOptParams
{
    bool enable = false;
    float corner_sigma = 2.0f;
    int min_inlier_corners = 3;  // 每相机每 Tag：至少 N/4 角点内点才保留该观测
    bool fixed_only = true;      // 第一版仅 FIXED_ANCHOR
    bool verbose = false;

    // 从 settings yaml 读取 Tag.pose_optimization / Tag.corner_sigma 等
    static TagPoseOptParams FromSettings(const std::string &settings_path);
};

// Local BA 中 Tag 刚体位姿 + 四角点重投影（独立于 Tracking 的 TagPoseOptParams）
struct TagLocalBAParams
{
    bool enable = false;             // Tag.local_ba.enable，默认关闭
    bool optimize_active = true;     // 1=ACTIVE 可优化；0=仅 FIXED_ANCHOR 入图
    float corner_sigma = 2.0f;
    int min_inlier_corners = 3;
    bool verbose = false;

    static TagLocalBAParams FromSettings(const std::string &settings_path);
};

// Pose-opt / Local BA 组级外点：同一 KF（或当前帧）× Tag × 相机
struct TagGroupOutlier
{
    unsigned long kf_id = 0;
    int tag_id = -1;
    CameraId camera = CameraId::LEFT_OR_MONO;
    int n_inlier_corners = 0;
    int n_corners = 0;
    bool is_opt_outlier = false;
};

struct TagLocalBAStats
{
    int num_tags = 0;
    int num_tag_edges = 0;
    int num_tag_outliers = 0;
    std::vector<int> outlier_tag_ids;
    std::vector<TagGroupOutlier> tag_groups;
};

// Local BA 无副作用求解结果：用于 ORB-only / fused 双分支对照。
// commit=false 时只填这个结构，不写地图。
struct LocalBAShadowResult
{
    std::vector<KeyFrame*> local_kfs;
    std::vector<KeyFrame*> fixed_kfs;
    std::vector<MapPoint*> local_mps;
    std::map<unsigned long, Sophus::SE3f> local_kf_poses;
    int num_tags = 0;
    int num_tag_edges = 0;
    int num_tag_outliers = 0;
    int num_opt_kf = 0;
    int num_fixed_kf = 0;
    int num_mps = 0;
    double tag_rmse_before = -1.0;
    double tag_rmse_after = -1.0;
    bool solved = false;
    bool forced_common_gauge = false;
    std::vector<int> outlier_tag_ids;
    std::vector<TagGroupOutlier> tag_groups;
};

// 在进入 PoseOptimization 前查询 MapTag，生成与当前帧观测对齐的约束快照。
// 只使用检测有效观测（IsDetectValid）；忽略 is_opt_outlier。
// 联合 PoseOptimization 入口会 ResetOptOutliers，同帧可再次加入 Tag。
std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params);

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
