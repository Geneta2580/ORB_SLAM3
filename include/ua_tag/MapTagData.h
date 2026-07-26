#ifndef UA_TAG_MAP_TAG_DATA_H
#define UA_TAG_MAP_TAG_DATA_H

#include "ua_tag/TagObservation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {
namespace tag {

// Tag 在地图中的全局位姿是否已确定
enum class MapTagState : std::uint8_t
{
    UNFIXED = 0,  // 尚未确定全局位姿
    FIXED = 1     // 全局位姿已确定
};

// 单个 Tag 在 Tag-map 中的数据：全局位姿 + 各帧上的观测
class MapTagData
{
public:
    // 观测所属帧 ID，统一使用 Frame::mnId
    using TagKeyFrameId = unsigned long;
    using Observations = std::vector<TagObservation>;
    // Frame::mnId -> 该帧上该 Tag 的观测（可含左右目多条）
    using ObservationMap = std::map<TagKeyFrameId, Observations>;

    int tag_id = -1;

    // Tag local -> Tag-map world（将 Tag 坐标系点变换到 Tag 地图世界系）
    Sophus::SE3f T_wt;
    bool has_pose = false;

    // 世界系下 Tag 四角点 3D 坐标（顺序与 IPPE_SQUARE / BuildSquareObjectPoints 一致）
    std::array<Eigen::Vector3f, 4> corners_world{};
    bool has_corners_world = false;

    MapTagState state = MapTagState::UNFIXED;

    ObservationMap observations;

    void SetPose(const Sophus::SE3f &T_wt_);
    const Sophus::SE3f &GetPose() const noexcept { return T_wt; }
    bool HasPose() const noexcept { return has_pose; }

    const std::array<Eigen::Vector3f, 4> &GetWorldCorners() const noexcept
    {
        return corners_world;
    }
    bool HasWorldCorners() const noexcept { return has_corners_world; }

    // 用当前 T_wt 与 tag_size（米）更新 corners_world：
    // P_w = T_wt * P_t，P_t 为 Tag 坐标系下固定正方形角点
    bool UpdateWorldCorners(double tag_size);

    void SetState(MapTagState state_) noexcept { state = state_; }
    MapTagState GetState() const noexcept { return state; }
    bool IsFixed() const noexcept { return state == MapTagState::FIXED; }

    // 向指定 TagKeyFrameId（Frame::mnId）追加一条观测（同一帧可有多条，如左右目）
    void AddObservation(TagKeyFrameId tag_kf_id, TagObservation observation);

    // 覆盖指定 TagKeyFrameId 的全部观测
    void SetObservations(TagKeyFrameId tag_kf_id, Observations obs);

    void EraseObservations(TagKeyFrameId tag_kf_id);

    const Observations *FindObservations(TagKeyFrameId tag_kf_id) const noexcept;
    Observations *FindObservations(TagKeyFrameId tag_kf_id) noexcept;

    bool HasTagKeyFrame(TagKeyFrameId tag_kf_id) const noexcept;

    void ClearObservations();

    bool Empty() const noexcept { return observations.empty(); }
    std::size_t NumTagKeyFrames() const noexcept { return observations.size(); }
    std::size_t NumObservations() const noexcept;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
