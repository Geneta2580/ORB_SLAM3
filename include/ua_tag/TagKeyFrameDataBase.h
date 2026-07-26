#ifndef UA_TAG_TAG_KEYFRAME_DATABASE_H
#define UA_TAG_TAG_KEYFRAME_DATABASE_H

#include "Frame.h"

#include <cstddef>
#include <vector>

namespace ORB_SLAM3 {
namespace tag {

// 轻量 Tag 关键帧库：以 Frame 为元素（含时间戳 / Tag 位姿 / TagFrameData 观测）
// 线程安全由外层 TagMap::mutex_ 负责
class TagKeyFrameDataBase
{
public:
    using Container = std::vector<Frame>;

    TagKeyFrameDataBase() = default;

    // 将 Frame 拷贝加入集合（需已带 mTagFrameData，建议已 SetTagPose）
    void AddKeyFrame(const Frame &frame);

    void Clear();

    bool Empty() const noexcept { return frames_.empty(); }
    std::size_t Size() const noexcept { return frames_.size(); }

    Frame &At(std::size_t idx) { return frames_.at(idx); }
    const Frame &At(std::size_t idx) const { return frames_.at(idx); }

    Frame &Front() { return frames_.front(); }
    const Frame &Front() const { return frames_.front(); }
    Frame &Back() { return frames_.back(); }
    const Frame &Back() const { return frames_.back(); }

    // 按 Frame::mnId 查找；未找到返回 nullptr
    Frame *FindByFrameId(unsigned long frame_id);
    const Frame *FindByFrameId(unsigned long frame_id) const;

    Container &GetAll() noexcept { return frames_; }
    const Container &GetAll() const noexcept { return frames_; }

private:
    Container frames_;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
