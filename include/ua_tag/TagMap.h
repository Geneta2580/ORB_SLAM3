#ifndef UA_TAG_MAP_H
#define UA_TAG_MAP_H

#include "ua_tag/MapTagData.h"
#include "ua_tag/TagInitializer.h"
#include "ua_tag/TagKeyFrameDataBase.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ORB_SLAM3 {

class Frame;

namespace tag {

enum class TagMapState : std::uint8_t
{
    NOT_INITIALIZED = 0,
    INITIALIZED = 1
};

class TagMap
{
public:
    using MapTagPtr = std::shared_ptr<MapTagData>;
    using TagContainer = std::unordered_map<int, MapTagPtr>;

public:
    TagMap() = default;

    // 从 settings yaml 读取 Tag.verbose、Tag.size
    explicit TagMap(const std::string &settingsFile);

    // TagMap 初始化接口，一次性提交所有结果
    bool Initialize(const TagInitializer::Result &result);

    bool IsInitialized() const noexcept;

    // 操作 MapTagData 相关数据接口
    MapTagPtr GetTag(int tag_id) const;

    MapTagPtr GetOrCreateTag(int tag_id);

    bool HasTag(int tag_id) const;

    void AddTag(const MapTagPtr &map_tag);

    void EraseTag(int tag_id);

    std::vector<MapTagPtr> GetAllTags() const;

    std::size_t NumTags() const;

    // 注册 Frame 观测到各 MapTag，并将 Frame 拷贝加入 TagKeyFrameDataBase
    void AddKeyFrame(const Frame &frame);

    TagKeyFrameDataBase &GetKeyFrameDataBase() noexcept { return keyframe_db_; }
    const TagKeyFrameDataBase &GetKeyFrameDataBase() const noexcept
    {
        return keyframe_db_;
    }

    std::size_t NumKeyFrames() const;

    // 将 Frame 上的 Tag 观测注册到 TagMap（不写入关键帧库）。
    // 若观测无歧义且帧有 TagPose，且该 MapTag 尚未 FIXED：
    //   T_wt = Twc * T_ct，SetPose + SetState(FIXED) + UpdateWorldCorners
    void RegisterFrameObservations(const Frame &frame);

    void EraseFrameObservations(unsigned long frame_id);

    // 显式重置整个 TagMap
    void Reset();

private:
    TagContainer tags_;
    TagKeyFrameDataBase keyframe_db_;

    TagMapState state_ = TagMapState::NOT_INITIALIZED;

    // 从 settings yaml 的 Tag.verbose 读取；为 true 时打印初始化日志
    bool mbVerbose = false;

    // 从 settings yaml 的 Tag.size 读取（米），用于更新世界系角点
    double mTagSize = 0.16;

    mutable std::mutex mutex_;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
