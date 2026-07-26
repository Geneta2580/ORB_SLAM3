#include "ua_tag/TagMap.h"

#include "Frame.h"

#include <iostream>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

TagMap::TagMap(const std::string &settingsFile)
{
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return;

    cv::FileNode node = fs["Tag.verbose"];
    if(!node.empty())
        mbVerbose = static_cast<int>(node) != 0;

    node = fs["Tag.size"];
    if(!node.empty())
        mTagSize = static_cast<double>(node);
}

bool TagMap::Initialize(const TagInitializer::Result &result)
{
    std::unique_lock<std::mutex> lock(mutex_);

    if(state_ == TagMapState::INITIALIZED)
        return false;

    if(result.tags.empty())
        return false;

    // 注册初始化结果中的 MapTag（含位姿与特殊 ID 观测）
    tags_.clear();
    keyframe_db_.Clear();
    for(const auto &kv : result.tags)
    {
        if(!kv.second)
            continue;
        tags_[kv.first] = kv.second;
    }

    if(tags_.empty())
        return false;

    // 用 T_wt 与固定 tag_size 更新各 Tag 世界系四角点
    for(auto &kv : tags_)
    {
        if(kv.second)
            kv.second->UpdateWorldCorners(mTagSize);
    }

    // 观测注册完成后，将初始化 Frame 写入 TagKeyFrameDataBase
    for(const Frame &kf : result.keyframes)
        keyframe_db_.AddKeyFrame(kf);

    state_ = TagMapState::INITIALIZED;

    if(mbVerbose)
    {
        std::cout << "[TagMap] Initialize succeeded"
                  << " num_tags=" << tags_.size()
                  << " num_keyframes=" << keyframe_db_.Size()
                  << " tag_ids=[";
        bool first = true;
        for(const auto &kv : tags_)
        {
            if(!first)
                std::cout << ", ";
            std::cout << kv.first;
            first = false;
        }
        std::cout << "]" << std::endl;

        const bool is_identity =
            result.Tcw_current.translation().squaredNorm() < 1e-12f &&
            (result.Tcw_current.so3().matrix() - Eigen::Matrix3f::Identity())
                    .squaredNorm() < 1e-12f;
        std::cout << "[TagMap] init frame Tcw ("
                  << (is_identity ? "single-frame I" : "two-frame T_21")
                  << "):\n"
                  << result.Tcw_current.matrix() << std::endl;
    }

    return true;
}

bool TagMap::IsInitialized() const noexcept
{
    std::unique_lock<std::mutex> lock(mutex_);
    return state_ == TagMapState::INITIALIZED;
}

TagMap::MapTagPtr TagMap::GetTag(int tag_id) const
{
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = tags_.find(tag_id);
    return (it == tags_.end()) ? nullptr : it->second;
}

TagMap::MapTagPtr TagMap::GetOrCreateTag(int tag_id)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = tags_.find(tag_id);
    if(it != tags_.end())
        return it->second;

    MapTagPtr map_tag = std::make_shared<MapTagData>();
    map_tag->tag_id = tag_id;
    tags_.emplace(tag_id, map_tag);
    return map_tag;
}

bool TagMap::HasTag(int tag_id) const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return tags_.find(tag_id) != tags_.end();
}

void TagMap::AddTag(const MapTagPtr &map_tag)
{
    if(!map_tag)
        return;

    std::unique_lock<std::mutex> lock(mutex_);
    tags_[map_tag->tag_id] = map_tag;
}

void TagMap::EraseTag(int tag_id)
{
    std::unique_lock<std::mutex> lock(mutex_);
    tags_.erase(tag_id);
}

std::vector<TagMap::MapTagPtr> TagMap::GetAllTags() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<MapTagPtr> out;
    out.reserve(tags_.size());
    for(const auto &kv : tags_)
        out.push_back(kv.second);
    return out;
}

std::size_t TagMap::NumTags() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return tags_.size();
}

void TagMap::AddKeyFrame(const Frame &frame)
{
    RegisterFrameObservations(frame);

    std::unique_lock<std::mutex> lock(mutex_);
    keyframe_db_.AddKeyFrame(frame);
}

std::size_t TagMap::NumKeyFrames() const
{
    std::unique_lock<std::mutex> lock(mutex_);
    return keyframe_db_.Size();
}

void TagMap::RegisterFrameObservations(const Frame &frame)
{
    const auto frame_id = frame.mnId;
    const tag::TagFrameData &frame_data = frame.mTagFrameData;

    if(frame_data.Empty())
        return;

    std::unique_lock<std::mutex> lock(mutex_);

    // 无歧义 IPPE + 当前帧 TagPose(Tcw)：T_wt = Twc * T_ct
    // 仅左目/单目；已 FIXED 的 MapTag 不覆盖
    auto try_fix_from_obs = [&](const MapTagPtr &map_tag, const TagObservation &obs) {
        if(!map_tag || map_tag->IsFixed())
            return;
        if(obs.camera_id != CameraId::LEFT_OR_MONO)
            return;
        if(obs.is_outlier || !obs.IsAmbiguityResolved())
            return;
        if(!frame.HasTagPose())
            return;

        const TagPoseCandidate *sel = obs.pose_estimate->Selected();
        if(!sel)
            return;

        const Sophus::SE3f T_wt = frame.GetTagPose().inverse() * sel->T_ct;
        map_tag->SetPose(T_wt);
        map_tag->SetState(MapTagState::FIXED);
        map_tag->UpdateWorldCorners(mTagSize);

        if(mbVerbose)
        {
            std::cout << "[TagMap] fix MapTag from unambiguous KF obs"
                      << " tag_id=" << map_tag->tag_id
                      << " frame_id=" << frame_id << std::endl;
        }
    };

    auto register_obs = [&](const TagObservation &obs) {
        MapTagPtr map_tag;
        auto it = tags_.find(obs.tag_id);
        if(it != tags_.end())
        {
            map_tag = it->second;
        }
        else
        {
            map_tag = std::make_shared<MapTagData>();
            map_tag->tag_id = obs.tag_id;
            tags_.emplace(obs.tag_id, map_tag);
        }
        map_tag->AddObservation(frame_id, obs);
        try_fix_from_obs(map_tag, obs);
    };

    for(const TagObservation &obs : frame_data.left)
        register_obs(obs);
    for(const TagObservation &obs : frame_data.right)
        register_obs(obs);
}

void TagMap::EraseFrameObservations(unsigned long frame_id)
{
    std::unique_lock<std::mutex> lock(mutex_);
    for(auto &kv : tags_)
    {
        if(kv.second)
            kv.second->EraseObservations(frame_id);
    }
}

void TagMap::Reset()
{
    std::unique_lock<std::mutex> lock(mutex_);
    tags_.clear();
    keyframe_db_.Clear();
    state_ = TagMapState::NOT_INITIALIZED;
}

}  // namespace tag
}  // namespace ORB_SLAM3
