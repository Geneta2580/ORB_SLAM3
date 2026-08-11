#ifndef UA_TAG_TAG_POSE_CONSTRAINTS_H
#define UA_TAG_TAG_POSE_CONSTRAINTS_H

#include "ua_tag/TagObservation.h"

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace ORB_SLAM3 {

class Frame;
class Map;

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

struct TagLocalBAStats
{
    int num_tags = 0;
    int num_tag_edges = 0;
    int num_tag_outliers = 0;
};

// 在进入 PoseOptimization 前查询 MapTag，生成与当前帧观测对齐的约束快照。
// 只使用检测有效观测（IsDetectValid）；忽略 is_opt_outlier。
// 联合 PoseOptimization 入口会 ResetOptOutliers，同帧可再次加入 Tag。
std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params);

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
