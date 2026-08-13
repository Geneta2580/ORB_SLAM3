#ifndef UA_TAG_TAG_EVAL_H
#define UA_TAG_TAG_EVAL_H

#include "ua_tag/TagPoseConstraints.h"

#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <sophus/se3.hpp>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {

class Frame;
class KeyFrame;
class Tracking;

namespace tag {

struct FramePoseOptState
{
    bool has_pose = false;
    Sophus::SE3f Tcw;
    std::vector<bool> mvbOutlier;
    struct TagFlags
    {
        bool is_opt_outlier = false;
        std::array<bool, 4> corner_outliers{{false, false, false, false}};
    };
    std::vector<TagFlags> left;
    std::vector<TagFlags> right;
};

// 单次运行双解对照：Pose-only / Local BA 的 ORB-only vs ORB+Tag，相对真值签过名的误差变化。
class TagEval
{
public:
    static std::unique_ptr<TagEval> FromSettings(const std::string &settings_path);

    bool Enabled() const { return mbEnable; }

    // 每个有 Tag 约束的 Tracking 帧都做 pose-only 双解。
    bool WantPoseShadow() const { return mbEnable; }

    // Tag 窗口 Local BA：每 lba_every_n 次抽样一次双解。
    bool WantLbaShadow();

    // 在 Tag 地图初始化前缓存轨迹，用于一次公共 SE3 对齐（ORB/fused 共用）。
    void ObservePoseForAlignment(double timestamp, const Sophus::SE3f &Tcw,
                                 bool tag_map_ready);

    void LogPoseShadow(const Frame &frame,
                       const std::vector<TagPoseConstraint> &tagSnap,
                       const Sophus::SE3f &Tcw_orb,
                       const Sophus::SE3f &Tcw_fused,
                       int orb_inliers,
                       int tag_corner_inliers,
                       float orb_rmse_at_orb,
                       float orb_rmse_at_fused,
                       float tag_rmse_at_orb,
                       float tag_rmse_at_fused);

    void LogLbaShadow(KeyFrame *pCurKF,
                      const LocalBAShadowResult &orb,
                      const LocalBAShadowResult &fused,
                      Tracking *pTracker);

    // 非双解 Local BA：只写组级外点明细（tag_eval_lba_tag.csv）。
    void LogLbaTagGroups(KeyFrame *pCurKF, const std::vector<TagGroupOutlier> &groups);

    static void SaveFramePoseState(const Frame &frame, FramePoseOptState &state);
    static void RestoreFramePoseState(Frame &frame, const FramePoseOptState &state);
    static int CountTagCornerInliers(const Frame &frame);
    static float ComputeOrbRmse(const Frame &frame);
    static float ComputeTagRmse(const Frame &frame,
                                const std::vector<TagPoseConstraint> &tagSnap,
                                bool use_opt_inliers);

private:
    enum class AlignMode
    {
        None,
        Yaml,
        PreTag
    };

    enum class GtFrame
    {
        Camera,
        Imu
    };

    bool OpenLogs();
    bool LoadGroundTruth(const std::string &path);
    static bool MatToSE3(const cv::Mat &T, Sophus::SE3f &out);
    bool InterpolateGtTwc(double timestamp, Sophus::SE3f &Twc_gt) const;
    Sophus::SE3f AlignEstTwc(const Sophus::SE3f &Twc_est) const;
    bool ComputeGtErrors(const Sophus::SE3f &Tcw_est, double timestamp,
                         float &trans_m, float &rot_deg) const;
    void TryFreezePreTagAlignment();
    static float Percentile(std::vector<float> values, float q);
    static Sophus::SE3f ReconstructTcw(const Sophus::SE3f &Tcr, KeyFrame *pKF,
                                       const std::map<unsigned long, Sophus::SE3f> &override_poses);

    bool mbEnable = false;
    std::string mOutDir = "tag_export";
    std::string mGtFile;
    GtFrame mGtFrame = GtFrame::Camera;
    bool mbGtWorldFromSensor = true;
    double mGtTimeScale = 1.0;
    Sophus::SE3f mT_b_c;  // p_imu = T_b_c * p_cam；仅 gt_frame=imu 时使用
    bool mbHasTbc = false;
    AlignMode mAlignMode = AlignMode::PreTag;
    int mLbaEveryN = 1;
    float mInducedTransTh = 0.01f;
    float mInducedRotThDeg = 0.5f;
    float mDeltaTransTh = 0.01f;
    float mDeltaRotThDeg = 0.5f;
    float mGtMaxDt = 0.05f;

    bool mbGtLoaded = false;
    std::vector<double> mvGtTime;
    std::vector<Sophus::SE3f> mvGtTwc;

    bool mbAligned = false;
    Sophus::SE3f mT_gt_slam;  // Twc_gt ≈ T_gt_slam * Twc_slam
    std::vector<double> mvAlignTime;
    std::vector<Sophus::SE3f> mvAlignTwc;

    int mnLbaTagWindows = 0;

    void WriteLbaTagGroups(KeyFrame *pCurKF, const std::vector<TagGroupOutlier> &groups);

    mutable std::mutex mMutex;
    std::ofstream mPoseFile;
    std::ofstream mPoseTagFile;
    std::ofstream mLbaFile;
    std::ofstream mLbaKfFile;
    std::ofstream mLbaTagFile;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
