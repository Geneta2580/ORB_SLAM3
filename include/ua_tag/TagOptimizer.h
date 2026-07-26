#ifndef UA_TAG_TAG_OPTIMIZER_H
#define UA_TAG_TAG_OPTIMIZER_H

#include <optional>

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Frame;

namespace tag {

class TagMap;

// Tag 优化 API（无独立线程；由 Tracking / TagTracker 同步调用）
class TagOptimizer
{
public:
    TagOptimizer() = delete;

    // Motion-only BA（GTSAM）：
    //   1) FIXED Tag 世界系角点作为 Point3 状态，NonlinearEquality 钉死
    //   2) 去畸变角点 + GenericProjectionFactor<Pose3,Point3,Cal3_S2>（Huber）
    //   3) 优化后按 chi2 标外点
    //   4) 仅内点、去 Huber 再优化一轮
    // 初值优先 Tcw_pred，否则用 frame 已有 TagPose；成功则用 BA 结果覆盖 SetTagPose
    // 失败不清除 frame 上已有 TagPose（保留匀速预测兜底）
    // verbose：失败打印 step 日志；成功打印 frame_id / n_inliers / n_covis_tags
    static void PoseOptimization(Frame &frame,
                                 const TagMap &tag_map,
                                 const std::optional<Sophus::SE3f> &Tcw_pred = std::nullopt,
                                 bool verbose = false);
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
