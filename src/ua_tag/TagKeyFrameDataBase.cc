#include "ua_tag/TagKeyFrameDataBase.h"

namespace ORB_SLAM3 {
namespace tag {

void TagKeyFrameDataBase::AddKeyFrame(const Frame &frame)
{
    frames_.push_back(frame);
}

void TagKeyFrameDataBase::Clear()
{
    frames_.clear();
}

Frame *TagKeyFrameDataBase::FindByFrameId(unsigned long frame_id)
{
    for(Frame &f : frames_)
    {
        if(f.mnId == frame_id)
            return &f;
    }
    return nullptr;
}

const Frame *TagKeyFrameDataBase::FindByFrameId(unsigned long frame_id) const
{
    for(const Frame &f : frames_)
    {
        if(f.mnId == frame_id)
            return &f;
    }
    return nullptr;
}

}  // namespace tag
}  // namespace ORB_SLAM3
