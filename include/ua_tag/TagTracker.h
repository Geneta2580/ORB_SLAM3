#ifndef UA_TAG_TAG_TRACKER_H
#define UA_TAG_TAG_TRACKER_H

#include <memory>
#include <optional>
#include <string>

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Frame;

namespace tag {

class TagMap;
class TagMapExporter;

class TagTracker
{
public:
    TagTracker();

    // 从 settings yaml 读取 Tag.verbose、Tag.export_dir 等
    explicit TagTracker(const std::string &settingsFile);
    ~TagTracker();

    TagTracker(const TagTracker &) = delete;
    TagTracker &operator=(const TagTracker &) = delete;

    // 在已有 TagMap 上跟踪当前帧
    bool Track(Frame &frame, TagMap &tag_map);

    // Tag 跟踪丢失后的重定位
    bool Relocalize(Frame &frame, TagMap &tag_map);

    // 初始化成功后写入首帧位姿缓存（尚无速度）
    void SeedLastPose(const Sophus::SE3f &Tcw);

    // 清空位姿/速度/参考关键帧临时缓存（跟踪失败或重置时调用）
    void ClearMotionCache();

    // 若开启导出：记录当前帧 TagPose 轨迹
    void LogCameraPose(const Frame &frame);

    // 若开启导出：写出 TagMap 四角点，并关闭轨迹文件
    void SaveExports(const TagMap &tag_map);

    long unsigned int GetReferenceKeyFrameId() const noexcept { return mnReferenceKFId; }
    bool HasReferenceKeyFrame() const noexcept { return mbHasReferenceKF; }

    // 是否启用条件 3：与参考 KF 的 baseline 判定
    void SetCheckRefKFBaseline(bool enable) noexcept { mbCheckRefKFBaseline = enable; }
    bool GetCheckRefKFBaseline() const noexcept { return mbCheckRefKFBaseline; }

    void SetMaxVisibleFramesPerMarker(int n) noexcept { mnMaxVisibleFramesPerMarker = n; }
    void SetMinBaseLine(float meters) noexcept { mMinBaseLine = meters; }

private:
    // 用缓存的匀速模型预测当前帧位姿；无效时返回 nullopt（由 Track 写入 Frame.TagPose）
    std::optional<Sophus::SE3f> PredictPoseWithMotionModel() const;

    // 为当前帧选择参考 Tag 关键帧：取与当前帧共视 Tag 数最高的历史 KF
    void SelectReferenceKeyFrame(const Frame &frame, TagMap &tag_map);

    // 关键帧判定（任一条件满足即为关键帧）：
    // 1) 当前帧观测到 TagMap 中尚不存在的 Tag
    // 2) 某 FIXED Tag：观测 KF 数 < maxVisibleFramesPerMarker，且相对已有观测 KF
    //    的相机平移距离 ≥ minBaseLine（为该 Tag 增加新视角）
    // 3) [可选] 与参考 KF 的 baseline > minBaseLine
    bool NeedNewKeyFrame(const Frame &frame, const TagMap &tag_map) const;

    // 用当前帧位姿更新上一帧 Pose 与 Velocity 缓存
    void UpdateMotionCache(const Sophus::SE3f &Tcw_current);

    // 上一帧相机位姿（Tag world -> Camera）
    Sophus::SE3f mLastTcw;
    bool mbHasLastTcw{false};

    // 上一帧相对运动：Tcw_cur * Twc_last（与 Tracking::mVelocity 同构）
    Sophus::SE3f mVelocity;
    bool mbHasVelocity{false};

    // 当前帧参考 Tag 关键帧（存 Frame::mnId）
    long unsigned int mnReferenceKFId{0};
    bool mbHasReferenceKF{false};

    // 每个 Tag 最多关联的观测关键帧数
    int mnMaxVisibleFramesPerMarker{10};
    // 新视角 / 参考 KF baseline 最小平移（米）
    float mMinBaseLine{0.07f};
    // 是否启用与参考 KF 的 baseline 关键帧条件
    bool mbCheckRefKFBaseline{false};

    // 从 settings yaml 的 Tag.verbose 读取；为 true 时打印跟踪/关键帧日志
    bool mbVerbose{false};

    std::unique_ptr<TagMapExporter> mpExporter;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
