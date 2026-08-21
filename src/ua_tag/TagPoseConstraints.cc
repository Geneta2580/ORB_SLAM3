#include "ua_tag/TagPoseConstraints.h"

#include "Frame.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagFrameData.h"

#include <iostream>
#include <unordered_set>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

bool ModuleEnabledFromSettings(const std::string &settings_path)
{
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return true;
    const cv::FileNode node = fs["Tag.enable"];
    if(node.empty())
        return true;
    return static_cast<int>(node) != 0;
}

TagPoseOptParams TagPoseOptParams::FromSettings(const std::string &settings_path)
{
    TagPoseOptParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;

    cv::FileNode node = fs["Tag.pose_optimization"];
    if(!node.empty())
    {
        params.enable = static_cast<int>(node) != 0;
        params.inertial_enable = params.enable;
    }

    // 兼容旧 yaml：仅写了 inertial 键时仍生效；与 pose_optimization 同时存在则以 pose_optimization 为准
    node = fs["Tag.inertial_pose_optimization"];
    if(!node.empty() && fs["Tag.pose_optimization"].empty())
        params.inertial_enable = static_cast<int>(node) != 0;

    node = fs["Tag.corner_sigma"];
    if(!node.empty())
        params.corner_sigma = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.min_inlier_corners"];
    if(!node.empty())
        params.min_inlier_corners = static_cast<int>(node);

    node = fs["Tag.fixed_only"];
    if(!node.empty())
        params.fixed_only = static_cast<int>(node) != 0;

    node = fs["Tag.verbose"];
    if(!node.empty())
        params.verbose = static_cast<int>(node) != 0;

    if(params.corner_sigma <= 1e-6f)
        params.corner_sigma = 2.0f;
    if(params.min_inlier_corners < 1)
        params.min_inlier_corners = 1;
    if(params.min_inlier_corners > 4)
        params.min_inlier_corners = 4;

    return params;
}

TagLocalBAParams TagLocalBAParams::FromSettings(const std::string &settings_path)
{
    TagLocalBAParams params;
    if(!ModuleEnabledFromSettings(settings_path))
        return params;

    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return params;

    cv::FileNode node = fs["Tag.local_ba.enable"];
    if(!node.empty())
        params.enable = static_cast<int>(node) != 0;

    node = fs["Tag.local_ba.optimize_active"];
    if(!node.empty())
        params.optimize_active = static_cast<int>(node) != 0;

    node = fs["Tag.local_ba.corner_sigma"];
    if(!node.empty())
        params.corner_sigma = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.local_ba.min_inlier_corners"];
    if(!node.empty())
        params.min_inlier_corners = static_cast<int>(node);

    node = fs["Tag.local_ba.verbose"];
    if(!node.empty())
        params.verbose = static_cast<int>(node) != 0;

    if(params.corner_sigma <= 1e-6f)
        params.corner_sigma = 2.0f;
    if(params.min_inlier_corners < 1)
        params.min_inlier_corners = 1;
    if(params.min_inlier_corners > 4)
        params.min_inlier_corners = 4;

    return params;
}

namespace {

bool ObservationEligible(const TagObservation &obs)
{
    // 仅看检测有效性；优化外点不得阻止同帧再次加入约束
    return obs.IsDetectValid();
}

bool CollectNeededTagIds(const Frame *pFrame, std::unordered_set<int> &ids)
{
    ids.clear();
    if(!pFrame)
        return false;

    for(const TagObservation &obs : pFrame->mTagFrameData.left)
    {
        if(ObservationEligible(obs))
            ids.insert(obs.tag_id);
    }
    for(const TagObservation &obs : pFrame->mTagFrameData.right)
    {
        if(ObservationEligible(obs))
            ids.insert(obs.tag_id);
    }
    return !ids.empty();
}

bool MapTagEligible(const MapTagData *pTag, const TagPoseOptParams &params,
                    bool inertial)
{
    if(!pTag || pTag->IsBad() || !pTag->HasWorldCorners())
        return false;
    if(inertial)
        return pTag->GetState() == MapTagState::ACTIVE;
    if(params.fixed_only)
        return pTag->GetState() == MapTagState::FIXED_ANCHOR;
    return pTag->GetState() == MapTagState::FIXED_ANCHOR ||
           pTag->GetState() == MapTagState::ACTIVE;
}

}  // namespace

std::vector<TagPoseConstraint> BuildTagMapSnapshot(
    Map *pMap, Frame *pFrame, const TagPoseOptParams &params, bool inertial)
{
    std::vector<TagPoseConstraint> out;
    const bool enabled = inertial ? params.inertial_enable : params.enable;
    if(!enabled || !pMap || !pFrame || !pMap->IsTagInitialized())
        return out;

    std::unordered_set<int> needed;
    if(!CollectNeededTagIds(pFrame, needed))
        return out;

    out.reserve(needed.size());
    for(int tag_id : needed)
    {
        const Map::MapTagPtr pTag = pMap->GetMapTag(tag_id);
        if(!MapTagEligible(pTag.get(), params, inertial))
            continue;

        const auto corners_f = pTag->GetWorldCorners();
        TagPoseConstraint c;
        c.tagId = tag_id;
        for(int i = 0; i < 4; ++i)
            c.worldCorners[i] = corners_f[i].cast<double>();
        out.push_back(c);
    }

    if(params.verbose)
    {
        std::cout << (inertial ? "[TagInertialPoseOpt]" : "[TagPoseOpt]")
                  << " snapshot tags=" << out.size()
                  << " needed=" << needed.size()
                  << " frame_id=" << pFrame->mnId << std::endl;
    }
    return out;
}

}  // namespace tag
}  // namespace ORB_SLAM3
