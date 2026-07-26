#ifndef UA_TAG_TAG_TRACKING_H
#define UA_TAG_TAG_TRACKING_H

#include <cstdint>

namespace ORB_SLAM3 {
namespace tag {

// 独立于 ORB-SLAM Tracking::eTrackingState
enum class TagTrackingState : std::uint8_t
{
    NOT_INITIALIZED = 0,
    OK = 1
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
