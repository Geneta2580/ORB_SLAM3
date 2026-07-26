#ifndef UA_TAG_TAG_OBSERVATION_H
#define UA_TAG_TAG_OBSERVATION_H

#include <array>
#include <cstdint>
#include <optional>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {
namespace tag {

enum class CameraId : std::uint8_t
{
    LEFT_OR_MONO = 0,
    RIGHT = 1
};

// 派生的位姿估计：Tag -> Camera（OpenCV / IPPE 约定）
struct TagPoseCandidate
{
    Sophus::SE3f T_ct;

    float reprojection_error = -1.0f;
    bool valid = false;
};

struct TagPoseEstimate
{
    // 仅存放有效 IPPE 解；IPPE_SQUARE 通常为 2 个
    std::array<TagPoseCandidate, 2> candidates{};

    // -1: 未消歧；0/1: 已选定的候选下标
    int selected_candidate = -1;

    float ambiguity_ratio = -1.0f;

    // 是否已消除 IPPE 二义性（selected_candidate 为有效下标且对应候选 valid）
    bool IsAmbiguityResolved() const noexcept
    {
        return Selected() != nullptr;
    }

    const TagPoseCandidate *Selected() const noexcept
    {
        if(selected_candidate < 0 || selected_candidate >= 2)
            return nullptr;

        const TagPoseCandidate &candidate = candidates[selected_candidate];
        return candidate.valid ? &candidate : nullptr;
    }
};

// 原始检测数据；pose_estimate 为空表示尚未做 IPPE / 消歧
struct TagObservation
{
    int tag_id = -1;
    CameraId camera_id = CameraId::LEFT_OR_MONO;

    // 原始检测角点（未去畸变，与输入图像像素一致）
    std::array<cv::Point2f, 4> corners_raw{};

    // 去畸变后的角点（EstimatePose 时写入；无畸变时等于 corners_raw）
    std::array<cv::Point2f, 4> corners_undistorted{};

    // 解码质量：bit 强度相对判决阈值的平均裕度，越大越可靠
    float decision_margin = 0.0f;
    // 解码时纠正的错误 bit 数，0 最好，越大误检风险越高
    int hamming = 0;

    // nullopt: 尚未估计；有值: 已计算候选（未必已消歧）
    std::optional<TagPoseEstimate> pose_estimate;

    bool is_outlier = false;

    // 是否已消除 IPPE 二义性
    bool IsAmbiguityResolved() const noexcept
    {
        return pose_estimate.has_value() && pose_estimate->IsAmbiguityResolved();
    }
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
