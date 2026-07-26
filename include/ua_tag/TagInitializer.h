#ifndef UA_TAG_TAG_INITIALIZER_H
#define UA_TAG_TAG_INITIALIZER_H

#include "Frame.h"
#include "ua_tag/MapTagData.h"

#include <array>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class GeometricCamera;

namespace tag {

class TagInitializer
{
public:
    using MapTagPtr = std::shared_ptr<MapTagData>;
    using TagContainer = std::unordered_map<int, MapTagPtr>;

    // Tag 地图初始化结果
    struct Result
    {
        // 初始化完成的 Tag 地图实体
        TagContainer tags;

        // 当前帧：Tag world -> Camera（单帧初始化时 world := 当前相机系，故为单位阵）
        Sophus::SE3f Tcw_current;

        // 初始化涉及的 Frame（已 SetTagPose；单帧 1 个，双帧为 ref + current）
        // 供 TagMap 写入 TagKeyFrameDataBase
        std::vector<Frame> keyframes;
    };

    // 双视图 IPPE 联合消歧结果
    struct TwoViewAmbiguityResult
    {
        Sophus::SE3f T_21;  // 参考帧(1) -> 当前帧(2) 的相对位姿
        float mean_reproj_error = std::numeric_limits<float>::infinity();

        // 各共视 Tag 消歧后选中的 IPPE 候选下标
        std::unordered_map<int, int> selected_ref_candidate;
        std::unordered_map<int, int> selected_cur_candidate;
        // 消歧后的 Tag->Camera 位姿
        std::unordered_map<int, Sophus::SE3f> T_ct_ref;
        std::unordered_map<int, Sophus::SE3f> T_ct_cur;

        bool valid = false;
    };

    // 双帧初始化所需的最少共同 Tag 数
    static constexpr std::size_t kMinCommonTagsForTwoFrame = 2;
    // 双帧初始化所需的平均角点像素位移相对图像宽度的最小比例
    static constexpr float kMinMeanPixelDispRatio = 0.02f;
    // 双帧最优解平均重投影误差上限（像素）
    static constexpr float kMaxMeanReprojErrorPx = 2.5f;
    // 双帧最优解相对位姿平移范数下限（米）
    static constexpr float kMinTranslationNormM = 0.07f;

public:
    // 从 settings yaml 读取 Tag.size、Tag.verbose
    explicit TagInitializer(const std::string &settingsFile);

    // 初始化流程：
    //   1) 始终先尝试单帧初始化
    //   2) 单帧失败时：
    //      - 无参考帧：将当前帧设为参考帧
    //      - 有参考帧且共同 Tag id >= 2：尝试双帧初始化
    //          * 双帧失败：保留原参考帧不动
    //      - 有参考帧但共同 Tag 过少：将参考帧切换为当前帧
    //   3) 成功则填充 result 并清空参考帧
    bool TryInitialize(Frame &frame, Result &result);

    void Clear();

    const std::deque<Frame> &GetInitFrames() const noexcept
    {
        return mInitFrames;
    }

private:
    // tag_id -> 已有两个有效 IPPE 候选的观测（不要求已消歧）
    using TagObsByIdMap = std::unordered_map<int, const TagObservation *>;
    // tag_id -> 无歧义且有效 IPPE 的观测
    using UnambiguousObsMap = std::unordered_map<int, const TagObservation *>;

    // 同一 tag_id 在两帧中的观测对（cur = 当前帧，ref = 参考帧）
    struct CommonTagPair
    {
        const TagObservation *cur = nullptr;
        const TagObservation *ref = nullptr;
    };
    using CommonTagObsMap = std::unordered_map<int, CommonTagPair>;

    // 非离群且 pose_estimate 中含两个有效 IPPE 候选
    static bool HasTwoValidIppeCandidates(const TagObservation &obs) noexcept;

    // 按 tag_id 收集观测；同 id 时左目优先于右目（要求两个有效 IPPE 候选）
    static void CollectObservationById(TagObsByIdMap &by_id, const TagObservation &obs);
    static TagObsByIdMap CollectObservationsById(const Frame &frame);

    // 将无歧义有效观测写入集合；同 tag_id 时左目优先于右目
    static void CollectUnambiguousObservation(UnambiguousObsMap &unambiguous,
                                              const TagObservation &obs);

    static UnambiguousObsMap CollectUnambiguousObservations(const Frame &frame);

    // 查找当前帧与参考帧之间共同 id 的 Tag 观测对
    static CommonTagObsMap FindCommonTagObservations(const Frame &frame,
                                                     const Frame &ref_frame);

    // 共同观测 Tag 的去畸变角点在参考帧与当前帧之间的平均欧式像素位移
    static float ComputeMeanCornerPixelDisplacement(const CommonTagObsMap &common_obs);

    // 用 GeometricCamera::project 计算 Tag 四角点相对观测的 RMSE（单位：像素）
    static float ComputeTagCornerReprojRmse(const Sophus::SE3f &T_ct,
                                            const std::array<cv::Point2f, 4> &corners_obs,
                                            GeometricCamera *pCamera,
                                            const std::vector<cv::Point3f> &object_pts);

    // 双视图 IPPE 联合消歧（独立流程）：
    //   1) 枚举全部共视 Tag 的 2x2 组合，按全场平均双向重投影选全局最优 T_21*
    //   2) 固定 T_21*，对每个共视 Tag 再遍历 4 种 IPPE 组合，取误差最小者消歧并回写
    bool ResolveTwoViewIppeAmbiguity(const CommonTagObsMap &common_obs,
                                     GeometricCamera *pCamera,
                                     TwoViewAmbiguityResult &out) const;

    // 单帧：当前帧中无歧义/有效 IPPE 的 Tag 数 >= 3 则成功
    // Tag world 固定为当前帧相机系；IPPE 的 T_ct 即为 T_wt，Tcw_current = I
    bool TryInitializeSingleFrame(Frame &frame, Result &result);

    // 双帧：当前帧 + 参考帧联合初始化（common_obs 由调用方预先计算）
    bool TryInitializeTwoFrames(Frame &frame, Frame &ref_frame,
                                const CommonTagObsMap &common_obs, Result &result);

    // 设置 / 切换双帧初始化的参考帧（仅保留一帧）
    void SetReferenceFrame(const Frame &frame);

    // 参考帧缓存（至多一帧）
    std::deque<Frame> mInitFrames;

    // 从 settings yaml 的 Tag.size 读取（米）
    double mTagSize = 0.16;

    // 从 settings yaml 的 Tag.verbose 读取；为 true 时打印初始化日志
    bool mbVerbose = false;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
