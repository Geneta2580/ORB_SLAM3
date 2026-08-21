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

// Tag.enable：总开关。缺省 true；0 时不检测、不建图、不进优化、不因 Tag 插 KF
bool ModuleEnabledFromSettings(const std::string &settings_path);

// 只读快照：优化期间不持有 Map/MapTag 锁
struct TagPoseConstraint
{
    int tagId = -1;
    std::array<Eigen::Vector3d, 4> worldCorners{};
};

struct TagPoseOptParams
{
    bool enable = false;                  // Tag.pose_optimization：PoseOptimization 与 PoseInertial* 共用
    bool inertial_enable = false;         // 与 enable 同步；仅旧 yaml 单独写 Tag.inertial_pose_optimization 时生效
    float corner_sigma = 2.0f;            // 纯视觉 / 惯性共用
    int min_inlier_corners = 3;           // 每相机每 Tag：至少 N/4 角点内点才保留；共用
    bool fixed_only = true;               // 仅纯视觉快照使用；惯性路径固定筛 ACTIVE
    bool verbose = false;                 // 共用

    // Tag.pose_optimization 同时控制纯视觉 PoseOptimization 与惯性 PoseInertial*
    // （两套代码分支，运行时只会走其中一条）。旧键 Tag.inertial_pose_optimization 仅作兼容。
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

// 在进入 PoseOptimization / PoseInertial* 前查询 MapTag，生成与当前帧观测对齐的约束快照。
// 只使用检测有效观测（IsDetectValid）；忽略 is_opt_outlier。
// 联合优化入口会 ResetOptOutliers，同帧可再次加入 Tag。
// inertial=false：看 params.enable；筛选受 params.fixed_only 控制。
// inertial=true：看 params.inertial_enable；仅 ACTIVE（不读 fixed_only）。
std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params,
    bool inertial = false);

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
