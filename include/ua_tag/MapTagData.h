#ifndef UA_TAG_MAP_TAG_DATA_H
#define UA_TAG_MAP_TAG_DATA_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class KeyFrame;
class Map;

namespace tag {

// MapTag 生命周期状态（转换逻辑后续阶段实现）
enum class MapTagState : std::uint8_t
{
    CANDIDATE = 0,   // 刚创建，观测不足
    ACTIVE,          // 正常地图实体，位姿可优化
    FIXED_ANCHOR,    // 固定的米制/世界坐标锚点
    BAD              // 已失效，等待从 Map 移除
};

/**
 * Map 中的单个 Tag 实体。
 *
 * 所有权（勿违反销毁顺序：先断 KF↔MapTag，再释放 Map 中的 shared_ptr）：
 *   Map                -> shared_ptr<MapTagData>   唯一持久化所有者
 *   KeyFrame           -> MapTagData*              非拥有
 *   MapTagData::mpMap  -> Map*                     非拥有
 *   MapTagData 观测    -> KeyFrame*                非拥有
 *
 * 唯一角点测量源：KeyFrame::mTagFrameData（本类只存左右观测索引）。
 */
class MapTagData
{
public:
    struct KeyFrameObservation
    {
        int leftIndex = -1;
        int rightIndex = -1;
    };

    using KeyFrameObservations = std::map<KeyFrame *, KeyFrameObservation>;

    MapTagData() = default;

    int Id() const;
    void SetId(int id);

    float GetTagSize() const;
    void SetTagSize(float tag_size);

    void SetPose(const Sophus::SE3f &T_wt);
    Sophus::SE3f GetPose() const;
    bool HasPose() const;

    // 按需计算：P_w = T_wt * P_t（需已有 pose 且 tag_size > 0）
    std::array<Eigen::Vector3f, 4> GetWorldCorners() const;
    bool HasWorldCorners() const;

    void SetState(MapTagState state);
    MapTagState GetState() const;
    bool IsFixed() const;  // FIXED_ANCHOR
    bool IsBad() const;

    Map *GetMap() const;

    bool IsInKeyFrame(KeyFrame *pKF) const;
    KeyFrameObservations GetObservations() const;
    std::size_t Observations() const;
    bool Empty() const { return Observations() == 0; }

private:
    friend class ORB_SLAM3::KeyFrame;
    friend class ORB_SLAM3::Map;

    void AddObservationInternal(KeyFrame *pKF, int leftIndex, int rightIndex);
    void EraseObservationInternal(KeyFrame *pKF);
    void SetMapInternal(Map *pMap);

    // mMutexPose: mTwt / mbHasPose / mTagSize
    mutable std::mutex mMutexPose;
    // mMutexFeatures: mObservations / mState；mpMap 由 Map 在持有 Map 锁时写入
    mutable std::mutex mMutexFeatures;

    int mnId = -1;
    float mTagSize = 0.0f;

    Sophus::SE3f mTwt;
    bool mbHasPose = false;

    MapTagState mState = MapTagState::CANDIDATE;
    KeyFrameObservations mObservations;

    Map *mpMap = nullptr;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
