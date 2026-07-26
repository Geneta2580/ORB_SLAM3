#include "ua_tag/MapTagData.h"

#include <utility>

namespace ORB_SLAM3 {
namespace tag {

void MapTagData::SetPose(const Sophus::SE3f &T_wt_)
{
    T_wt = T_wt_;
    has_pose = true;
}

bool MapTagData::UpdateWorldCorners(double tag_size)
{
    if(!has_pose || tag_size <= 0.0)
    {
        has_corners_world = false;
        return false;
    }

    // 与 OpenCV IPPE_SQUARE / ua_tag::BuildSquareObjectPoints 约定一致：
    // (-s/2,s/2,0), (s/2,s/2,0), (s/2,-s/2,0), (-s/2,-s/2,0)
    const float h = static_cast<float>(tag_size) * 0.5f;
    const Eigen::Vector3f pts_t[4] = {
        Eigen::Vector3f(-h,  h, 0.f),
        Eigen::Vector3f( h,  h, 0.f),
        Eigen::Vector3f( h, -h, 0.f),
        Eigen::Vector3f(-h, -h, 0.f)};

    for(int i = 0; i < 4; ++i)
        corners_world[i] = T_wt * pts_t[i];

    has_corners_world = true;
    return true;
}

void MapTagData::AddObservation(TagKeyFrameId tag_kf_id, TagObservation observation)
{
    observations[tag_kf_id].push_back(std::move(observation));
}

void MapTagData::SetObservations(TagKeyFrameId tag_kf_id, Observations obs)
{
    observations[tag_kf_id] = std::move(obs);
}

void MapTagData::EraseObservations(TagKeyFrameId tag_kf_id)
{
    observations.erase(tag_kf_id);
}

const MapTagData::Observations *MapTagData::FindObservations(TagKeyFrameId tag_kf_id) const noexcept
{
    const auto it = observations.find(tag_kf_id);
    return (it == observations.end()) ? nullptr : &it->second;
}

MapTagData::Observations *MapTagData::FindObservations(TagKeyFrameId tag_kf_id) noexcept
{
    auto it = observations.find(tag_kf_id);
    return (it == observations.end()) ? nullptr : &it->second;
}

bool MapTagData::HasTagKeyFrame(TagKeyFrameId tag_kf_id) const noexcept
{
    return observations.find(tag_kf_id) != observations.end();
}

void MapTagData::ClearObservations()
{
    observations.clear();
}

std::size_t MapTagData::NumObservations() const noexcept
{
    std::size_t n = 0;
    for(const auto &kv : observations)
        n += kv.second.size();
    return n;
}

}  // namespace tag
}  // namespace ORB_SLAM3
