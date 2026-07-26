#include "ua_tag/TagTracker.h"

#include "Frame.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagKeyFrameDataBase.h"
#include "ua_tag/TagMap.h"
#include "ua_tag/TagMapExporter.h"
#include "ua_tag/TagOptimizer.h"

#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

TagTracker::TagTracker() = default;

TagTracker::~TagTracker() = default;

TagTracker::TagTracker(const std::string &settingsFile)
{
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return;

    cv::FileNode node = fs["Tag.verbose"];
    if(!node.empty())
        mbVerbose = static_cast<int>(node) != 0;

    // Tag.export=1 时启用导出；目录默认 tag_export，可用 Tag.export_dir 覆盖
    bool enable_export = false;
    node = fs["Tag.export"];
    if(!node.empty())
        enable_export = static_cast<int>(node) != 0;

    std::string export_dir = "tag_export";
    node = fs["Tag.export_dir"];
    if(!node.empty())
        export_dir = static_cast<std::string>(node);

    if(enable_export)
        mpExporter = std::make_unique<TagMapExporter>(export_dir);
}

void TagTracker::LogCameraPose(const Frame &frame)
{
    if(!mpExporter || !mpExporter->IsEnabled() || !frame.HasTagPose())
        return;
    mpExporter->AppendCameraPose(frame.mnId, frame.mTimeStamp, frame.GetTagPose());
}

void TagTracker::SaveExports(const TagMap &tag_map)
{
    if(!mpExporter || !mpExporter->IsEnabled())
        return;
    mpExporter->SaveTagMapCorners(tag_map);
    mpExporter->CloseTrajectory();
}

void TagTracker::SeedLastPose(const Sophus::SE3f &Tcw)
{
    mLastTcw = Tcw;
    mbHasLastTcw = true;
    mbHasVelocity = false;
}

void TagTracker::ClearMotionCache()
{
    mbHasLastTcw = false;
    mbHasVelocity = false;
    mnReferenceKFId = 0;
    mbHasReferenceKF = false;
}

std::optional<Sophus::SE3f> TagTracker::PredictPoseWithMotionModel() const
{
    if(!mbHasVelocity || !mbHasLastTcw)
        return std::nullopt;

    return mVelocity * mLastTcw;
}

void TagTracker::SelectReferenceKeyFrame(const Frame &frame, TagMap &tag_map)
{
    mbHasReferenceKF = false;
    mnReferenceKFId = 0;

    // 当前帧观测到的 Tag id（左右目去重）
    std::unordered_set<int> cur_tag_ids;
    for(const TagObservation &obs : frame.mTagFrameData.left)
        cur_tag_ids.insert(obs.tag_id);
    for(const TagObservation &obs : frame.mTagFrameData.right)
        cur_tag_ids.insert(obs.tag_id);

    if(cur_tag_ids.empty())
        return;

    // 遍历当前帧 Tag 的历史共视 KF，累计每个 KF 与当前帧的共视 Tag 数
    std::unordered_map<unsigned long, int> covis_scores;
    for(const int tag_id : cur_tag_ids)
    {
        const auto map_tag = tag_map.GetTag(tag_id);
        if(!map_tag)
            continue;

        for(const auto &kv : map_tag->observations)
        {
            const unsigned long kf_id = kv.first;
            if(kf_id == frame.mnId)
                continue;
            ++covis_scores[kf_id];
        }
    }

    if(covis_scores.empty())
        return;

    const TagKeyFrameDataBase &kf_db = tag_map.GetKeyFrameDataBase();

    unsigned long best_id = 0;
    int best_score = -1;
    for(const auto &kv : covis_scores)
    {
        if(kv.second < best_score)
            continue;
        if(kv.second == best_score && kv.first <= best_id)
            continue;
        if(!kf_db.FindByFrameId(kv.first))
            continue;

        best_score = kv.second;
        best_id = kv.first;
    }

    if(best_score < 0)
        return;

    mnReferenceKFId = best_id;
    mbHasReferenceKF = true;
}

bool TagTracker::NeedNewKeyFrame(const Frame &frame, const TagMap &tag_map) const
{
    if(!frame.HasTagPose())
        return false;

    // 当前帧非外点观测 Tag（左右目去重）
    std::unordered_set<int> cur_tag_ids;
    auto collect = [&](const TagObservation &obs) {
        if(!obs.is_outlier)
            cur_tag_ids.insert(obs.tag_id);
    };
    for(const TagObservation &obs : frame.mTagFrameData.left)
        collect(obs);
    for(const TagObservation &obs : frame.mTagFrameData.right)
        collect(obs);

    if(cur_tag_ids.empty())
        return false;

    // 1) 当前帧观测存在 TagMap 中没有的 Tag
    for(const int tag_id : cur_tag_ids)
    {
        if(!tag_map.HasTag(tag_id))
            return true;
    }

    const TagKeyFrameDataBase &kf_db = tag_map.GetKeyFrameDataBase();
    const Eigen::Vector3f Ow_cur = frame.GetTagPose().inverse().translation();

    // 2) 已有有效全局 pose 的 Tag：观测 KF 数未满，且相对已有观测 KF 平移够大
    for(const int tag_id : cur_tag_ids)
    {
        const auto map_tag = tag_map.GetTag(tag_id);
        if(!map_tag || !map_tag->HasPose() || !map_tag->IsFixed())
            continue;

        if(static_cast<int>(map_tag->NumTagKeyFrames()) >= mnMaxVisibleFramesPerMarker)
            continue;

        float min_dist = std::numeric_limits<float>::infinity();
        bool has_valid_kf = false;
        for(const auto &kv : map_tag->observations)
        {
            const unsigned long kf_id = kv.first;
            if(kf_id == frame.mnId)
                continue;

            const Frame *pkf = kf_db.FindByFrameId(kf_id);
            if(!pkf || !pkf->HasTagPose())
                continue;

            const float dist =
                (Ow_cur - pkf->GetTagPose().inverse().translation()).norm();
            if(dist < min_dist)
                min_dist = dist;
            has_valid_kf = true;
        }

        // 尚无历史观测 KF：可直接作为该 Tag 的新视角
        if(!has_valid_kf || min_dist >= mMinBaseLine)
            return true;
    }

    // 3) [可选] 与参考 KF 的 baseline > minBaseLine
    if(mbCheckRefKFBaseline && mbHasReferenceKF)
    {
        const Frame *pref = kf_db.FindByFrameId(mnReferenceKFId);
        if(pref && pref->HasTagPose())
        {
            const float baseline =
                (Ow_cur - pref->GetTagPose().inverse().translation()).norm();
            if(baseline > mMinBaseLine)
                return true;
        }
    }

    return false;
}

void TagTracker::UpdateMotionCache(const Sophus::SE3f &Tcw_current)
{
    if(mbHasLastTcw)
    {
        mVelocity = Tcw_current * mLastTcw.inverse();
        mbHasVelocity = true;
    }

    mLastTcw = Tcw_current;
    mbHasLastTcw = true;
}

bool TagTracker::Track(Frame &frame, TagMap &tag_map)
{
    // BA / 兜底初值：有速度则匀速递推；否则用上一帧 pose（初始化后首帧尚无速度）
    const std::optional<Sophus::SE3f> Tcw_pred = PredictPoseWithMotionModel();
    if(Tcw_pred)
        frame.SetTagPose(*Tcw_pred);
    else if(mbHasLastTcw)
        frame.SetTagPose(mLastTcw);

    // 选择当前帧参考关键帧
    SelectReferenceKeyFrame(frame, tag_map);

    const std::optional<Sophus::SE3f> Tcw_init =
        frame.HasTagPose() ? std::optional<Sophus::SE3f>(frame.GetTagPose())
                           : std::nullopt;

    // Motion-only BA：成功则覆盖 TagPose；失败则保留上方初值兜底
    TagOptimizer::PoseOptimization(frame, tag_map, Tcw_init, mbVerbose);

    if(!frame.HasTagPose())
        return false;

    UpdateMotionCache(frame.GetTagPose());
    LogCameraPose(frame);

    const bool need_keyframe = NeedNewKeyFrame(frame, tag_map);
    if(need_keyframe)
        tag_map.AddKeyFrame(frame);

    if(mbVerbose)
    {
        std::cout << "[TagTracker] frame_id=" << frame.mnId
                  << " keyframe=" << (need_keyframe ? 1 : 0)
                  << " num_map_kf=" << tag_map.NumKeyFrames()
                  << " ref_kf_id=" << (mbHasReferenceKF ? mnReferenceKFId : 0)
                  << std::endl;
    }

    return true;
}

bool TagTracker::Relocalize(Frame &frame, TagMap &tag_map)
{
    (void)frame;
    (void)tag_map;
    // TODO: Tag 重定位
    return false;
}

}  // namespace tag
}  // namespace ORB_SLAM3
