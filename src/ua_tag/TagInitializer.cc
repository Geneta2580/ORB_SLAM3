#include "ua_tag/TagInitializer.h"

#include "GeometricCamera.h"
#include "ua_tag/AprilTagDetector.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

namespace ORB_SLAM3 {
namespace tag {

TagInitializer::TagInitializer(const std::string &settingsFile)
{
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return;

    cv::FileNode node = fs["Tag.size"];
    if(!node.empty())
        mTagSize = static_cast<double>(node);

    node = fs["Tag.verbose"];
    if(!node.empty())
        mbVerbose = static_cast<int>(node) != 0;
}

void TagInitializer::Clear()
{
    mInitFrames.clear();
}

void TagInitializer::SetReferenceFrame(const Frame &frame)
{
    if(mbVerbose)
        std::cout << "[TagInitializer] SetReferenceFrame frame_id=" << frame.mnId
                  << std::endl;

    mInitFrames.clear();
    mInitFrames.push_back(frame);
}

bool TagInitializer::HasTwoValidIppeCandidates(const TagObservation &obs) noexcept
{
    if(obs.is_outlier || !obs.pose_estimate.has_value())
        return false;

    const TagPoseEstimate &est = *obs.pose_estimate;
    return est.candidates[0].valid && est.candidates[1].valid;
}

// 按 tag_id 收集观测；同 id 时左目优先于右目（要求两个有效 IPPE 候选）
void TagInitializer::CollectObservationById(TagObsByIdMap &by_id,
                                            const TagObservation &obs)
{
    if(!HasTwoValidIppeCandidates(obs))
        return;

    auto it = by_id.find(obs.tag_id);
    if(it == by_id.end())
    {
        by_id.emplace(obs.tag_id, &obs);
        return;
    }

    // 已有观测时：左目优先于右目；同相机保留先写入的那条
    const TagObservation &prev = *it->second;
    if(prev.camera_id == CameraId::LEFT_OR_MONO)
        return;
    if(obs.camera_id == CameraId::LEFT_OR_MONO)
        it->second = &obs;
}

// 按 tag_id 收集参考帧和当前帧的多个观测；同 id 时左目优先于右目（不要求 IPPE 无歧义）
TagInitializer::TagObsByIdMap TagInitializer::CollectObservationsById(const Frame &frame)
{
    TagObsByIdMap by_id;
    for(const TagObservation &obs : frame.mTagFrameData.left)
        CollectObservationById(by_id, obs);
    for(const TagObservation &obs : frame.mTagFrameData.right)
        CollectObservationById(by_id, obs);
    return by_id;
}

// 按 tag_id 收集无歧义且有效的 IPPE 解的单个观测；同 id 时左目优先于右目
void TagInitializer::CollectUnambiguousObservation(UnambiguousObsMap &unambiguous,
                                                   const TagObservation &obs)
{
    // 无歧义且有效的 IPPE 解：已消歧选中，且候选 valid
    if(obs.is_outlier || !obs.pose_estimate.has_value() ||
       obs.pose_estimate->Selected() == nullptr)
        return;

    auto it = unambiguous.find(obs.tag_id);
    if(it == unambiguous.end())
    {
        unambiguous.emplace(obs.tag_id, &obs);
        return;
    }

    // 已有观测时：左目优先于右目；同相机保留先写入的那条
    const TagObservation &prev = *it->second;
    if(prev.camera_id == CameraId::LEFT_OR_MONO)
        return;
    if(obs.camera_id == CameraId::LEFT_OR_MONO)
        it->second = &obs;
}

// 按 tag_id 收集无歧义且有效的 IPPE 解的多个观测；同 id 时左目优先于右目
TagInitializer::UnambiguousObsMap TagInitializer::CollectUnambiguousObservations(
    const Frame &frame)
{
    UnambiguousObsMap unambiguous;
    for(const TagObservation &obs : frame.mTagFrameData.left)
        CollectUnambiguousObservation(unambiguous, obs);
    for(const TagObservation &obs : frame.mTagFrameData.right)
        CollectUnambiguousObservation(unambiguous, obs);
    return unambiguous;
}

TagInitializer::CommonTagObsMap TagInitializer::FindCommonTagObservations(
    const Frame &frame, const Frame &ref_frame)
{
    // 收集当前帧和参考帧的观测（以id为索引）
    const TagObsByIdMap cur_by_id = CollectObservationsById(frame);
    const TagObsByIdMap ref_by_id = CollectObservationsById(ref_frame);

    CommonTagObsMap common;
    for(const auto &kv : cur_by_id)
    {
        const auto ref_it = ref_by_id.find(kv.first);
        if(ref_it == ref_by_id.end())
            continue;

        CommonTagPair pair;
        pair.cur = kv.second;
        pair.ref = ref_it->second;
        common.emplace(kv.first, pair);
    }
    return common;
}

float TagInitializer::ComputeMeanCornerPixelDisplacement(const CommonTagObsMap &common_obs)
{
    double sum = 0.0;
    std::size_t count = 0;

    for(const auto &kv : common_obs)
    {
        const CommonTagPair &pair = kv.second;
        if(!pair.cur || !pair.ref)
            continue;

        const auto &corners_cur = pair.cur->corners_undistorted;
        const auto &corners_ref = pair.ref->corners_undistorted;

        for(int i = 0; i < 4; ++i)
        {
            const float dx = corners_cur[i].x - corners_ref[i].x;
            const float dy = corners_cur[i].y - corners_ref[i].y;
            sum += std::sqrt(static_cast<double>(dx * dx + dy * dy));
            ++count;
        }
    }

    if(count == 0)
        return 0.0f;

    return static_cast<float>(sum / static_cast<double>(count));
}

float TagInitializer::ComputeTagCornerReprojRmse(
    const Sophus::SE3f &T_ct,
    const std::array<cv::Point2f, 4> &corners_obs,
    GeometricCamera *pCamera,
    const std::vector<cv::Point3f> &object_pts)
{
    if(!pCamera || object_pts.size() != 4)
        return std::numeric_limits<float>::infinity();

    double sum_sq = 0.0;
    for(int i = 0; i < 4; ++i)
    {
        const Eigen::Vector3f P_t(object_pts[i].x, object_pts[i].y, object_pts[i].z);
        const Eigen::Vector3f P_c = T_ct * P_t;
        if(P_c.z() <= 0.0f)
            return std::numeric_limits<float>::infinity();

        // 复用 ORB-SLAM GeometricCamera::project（与 mvKeysUn / 去畸变像素一致）
        const cv::Point2f uv = pCamera->project(cv::Point3f(P_c.x(), P_c.y(), P_c.z()));
        const double dx = static_cast<double>(uv.x - corners_obs[i].x);
        const double dy = static_cast<double>(uv.y - corners_obs[i].y);
        sum_sq += dx * dx + dy * dy;
    }

    return static_cast<float>(std::sqrt(sum_sq / 4.0));
}

bool TagInitializer::ResolveTwoViewIppeAmbiguity(const CommonTagObsMap &common_obs,
                                                GeometricCamera *pCamera,
                                                TwoViewAmbiguityResult &out) const
{
    out = TwoViewAmbiguityResult{};
    if(!pCamera || mTagSize <= 0.0 || common_obs.empty())
        return false;

    std::vector<cv::Point3f> object_pts;
    ua_tag::BuildSquareObjectPoints(mTagSize, object_pts);

    // 固定 T_21 下，一对 (i_ref, i_cur) 的正向/反向重投影误差平均
    auto ComputePairReprojError =
        [&](const Sophus::SE3f &T_21, const Sophus::SE3f &T_12,
            const TagObservation &obs_ref, const TagObservation &obs_cur,
            int i_ref, int i_cur) -> float {
        const TagPoseEstimate &est_ref = *obs_ref.pose_estimate;
        const TagPoseEstimate &est_cur = *obs_cur.pose_estimate;
        if(!est_ref.candidates[i_ref].valid || !est_cur.candidates[i_cur].valid)
            return std::numeric_limits<float>::infinity();

        const Sophus::SE3f T_c2t_pred = T_21 * est_ref.candidates[i_ref].T_ct;
        const float err_12 = ComputeTagCornerReprojRmse(
            T_c2t_pred, obs_cur.corners_undistorted, pCamera, object_pts);

        const Sophus::SE3f T_c1t_pred = T_12 * est_cur.candidates[i_cur].T_ct;
        const float err_21 = ComputeTagCornerReprojRmse(
            T_c1t_pred, obs_ref.corners_undistorted, pCamera, object_pts);

        if(!std::isfinite(err_12) || !std::isfinite(err_21))
            return std::numeric_limits<float>::infinity();

        return 0.5f * (err_12 + err_21);
    };

    // 固定 T_21：每个共视 Tag 取 4 种组合中最小双向误差，再平均
    auto EvaluateMeanReprojError = [&](const Sophus::SE3f &T_21,
                                       float &mean_err) -> bool {
        const Sophus::SE3f T_12 = T_21.inverse();
        double sum_min_err = 0.0;
        std::size_t n_tags = 0;

        for(const auto &kv : common_obs)
        {
            const TagObservation &obs_ref = *kv.second.ref;
            const TagObservation &obs_cur = *kv.second.cur;

            float min_err = std::numeric_limits<float>::infinity();
            for(int i_ref = 0; i_ref < 2; ++i_ref)
            {
                for(int i_cur = 0; i_cur < 2; ++i_cur)
                {
                    const float err = ComputePairReprojError(
                        T_21, T_12, obs_ref, obs_cur, i_ref, i_cur);
                    if(err < min_err)
                        min_err = err;
                }
            }

            if(!std::isfinite(min_err))
                return false;

            sum_min_err += static_cast<double>(min_err);
            ++n_tags;
        }

        if(n_tags == 0)
            return false;

        mean_err = static_cast<float>(sum_min_err / static_cast<double>(n_tags));
        return true;
    };

    // 1) 在全部共视 Tag 的 2x2 组合上枚举 T_21，选全局平均重投影最小者，固定T_21
    float best_mean_err = std::numeric_limits<float>::infinity();
    Sophus::SE3f best_T21;

    for(const auto &kv : common_obs)
    {
        const TagPoseEstimate &est_ref = *kv.second.ref->pose_estimate;
        const TagPoseEstimate &est_cur = *kv.second.cur->pose_estimate;

        for(int i_ref = 0; i_ref < 2; ++i_ref)
        {
            if(!est_ref.candidates[i_ref].valid)
                continue;
            const Sophus::SE3f &T_c1t = est_ref.candidates[i_ref].T_ct;

            for(int i_cur = 0; i_cur < 2; ++i_cur)
            {
                if(!est_cur.candidates[i_cur].valid)
                    continue;
                const Sophus::SE3f &T_c2t = est_cur.candidates[i_cur].T_ct;
                const Sophus::SE3f T_21 = T_c2t * T_c1t.inverse();

                float mean_err = std::numeric_limits<float>::infinity();
                if(!EvaluateMeanReprojError(T_21, mean_err))
                    continue;

                if(mean_err < best_mean_err)
                {
                    best_mean_err = mean_err;
                    best_T21 = T_21;
                }
            }
        }
    }

    if(!std::isfinite(best_mean_err))
        return false;

    out.T_21 = best_T21;
    out.mean_reproj_error = best_mean_err;

    const float translation_norm = out.T_21.translation().norm();
    if(mbVerbose)
        std::cout << "[TagInitializer] best two-view solution"
                  << " mean_reproj_error=" << out.mean_reproj_error << " px"
                  << " (need < " << kMaxMeanReprojErrorPx << " px)"
                  << ", |t|=" << translation_norm << " m"
                  << " (need > " << kMinTranslationNormM << " m)"
                  << std::endl;

    if(out.mean_reproj_error >= kMaxMeanReprojErrorPx ||
       translation_norm <= kMinTranslationNormM)
    {
        if(mbVerbose)
            std::cout << "[TagInitializer] TryInitializeTwoFrames failed"
                      << " (best solution rejected"
                      << ", mean_reproj_error=" << out.mean_reproj_error << " px"
                      << ", |t|=" << translation_norm << " m)"
                      << std::endl;
        return false;
    }

    // 2) 固定全局最优 T_21*，对每个共视 Tag 遍历 4 种 IPPE 组合，
    //    取双向重投影之和最小者作为消歧结果并回写
    const Sophus::SE3f T_12 = out.T_21.inverse();
    for(const auto &kv : common_obs)
    {
        const int tag_id = kv.first;
        const TagObservation &obs_ref = *kv.second.ref;
        const TagObservation &obs_cur = *kv.second.cur;
        const TagPoseEstimate &est_ref = *obs_ref.pose_estimate;
        const TagPoseEstimate &est_cur = *obs_cur.pose_estimate;

        float best_err = std::numeric_limits<float>::infinity();
        int best_i_ref = -1;
        int best_i_cur = -1;

        for(int i_ref = 0; i_ref < 2; ++i_ref)
        {
            for(int i_cur = 0; i_cur < 2; ++i_cur)
            {
                const float err = ComputePairReprojError(
                    out.T_21, T_12, obs_ref, obs_cur, i_ref, i_cur);
                if(err < best_err)
                {
                    best_err = err;
                    best_i_ref = i_ref;
                    best_i_cur = i_cur;
                }
            }
        }

        if(best_i_ref < 0 || best_i_cur < 0 || !std::isfinite(best_err))
            continue;

        out.selected_ref_candidate[tag_id] = best_i_ref;
        out.selected_cur_candidate[tag_id] = best_i_cur;
        out.T_ct_ref.emplace(tag_id, est_ref.candidates[best_i_ref].T_ct);
        out.T_ct_cur.emplace(tag_id, est_cur.candidates[best_i_cur].T_ct);
    }

    out.valid = !out.T_ct_ref.empty();
    return out.valid;
}

bool TagInitializer::TryInitializeSingleFrame(Frame &frame, Result &result)
{
    constexpr std::size_t kMinUnambiguousTagsForSingleFrame = 3;

    if(mbVerbose)
        std::cout << "[TagInitializer] TryInitializeSingleFrame running"
                  << " (frame_id=" << frame.mnId << ")" << std::endl;

    // 收集无歧义的有效观测
    const UnambiguousObsMap unambiguous = CollectUnambiguousObservations(frame);

    if(mbVerbose)
        std::cout << "[TagInitializer] unambiguous valid observations: "
                  << unambiguous.size()
                  << " (need >= " << kMinUnambiguousTagsForSingleFrame << ")"
                  << std::endl;

    // 无歧义观测数不足
    if(unambiguous.size() < kMinUnambiguousTagsForSingleFrame)
    {
        if(mbVerbose)
            std::cout << "[TagInitializer] TryInitializeSingleFrame failed"
                      << " (insufficient unambiguous observations)" << std::endl;
        return false;
    }

    // Tag world := 当前帧相机坐标系 => Tcw = I，IPPE 的 T_ct 即为 T_wt
    Result out;
    out.Tcw_current = Sophus::SE3f();

    // 遍历无歧义观测，构建 Tag 初始化结果并写入观测（统一用 Frame::mnId）
    for(const auto &kv : unambiguous)
    {
        const int tag_id = kv.first;
        const TagObservation &obs = *kv.second;
        const Sophus::SE3f &T_ct = obs.pose_estimate->Selected()->T_ct;

        MapTagPtr map_tag = std::make_shared<MapTagData>();
        map_tag->tag_id = tag_id;
        map_tag->SetPose(T_ct);  // T_wt = T_ct（world = camera）
        map_tag->SetState(MapTagState::FIXED);
        map_tag->AddObservation(frame.mnId, obs);

        out.tags.emplace(tag_id, std::move(map_tag));
    }

    // 供 TagKeyFrameDataBase 注册：当前帧 + Tag 位姿
    Frame kf = frame;
    kf.SetTagPose(out.Tcw_current);
    out.keyframes.push_back(std::move(kf));

    result = std::move(out);

    if(mbVerbose)
        std::cout << "[TagInitializer] TryInitializeSingleFrame succeeded"
                  << " (tags=" << result.tags.size() << ")" << std::endl;

    return true;
}

bool TagInitializer::TryInitializeTwoFrames(Frame &frame, Frame &ref_frame,
                                            const CommonTagObsMap &common_obs,
                                            Result &result)
{
    if(mbVerbose)
        std::cout << "[TagInitializer] TryInitializeTwoFrames running"
                  << " (frame_id=" << frame.mnId
                  << ", ref_frame_id=" << ref_frame.mnId
                  << ", common_tags=" << common_obs.size() << ")" << std::endl;

    // 共同观测去畸变角点的平均像素位移
    const float mean_pixel_disp = ComputeMeanCornerPixelDisplacement(common_obs);

    if(mbVerbose)
        std::cout << "[TagInitializer] mean corner pixel displacement: "
                  << mean_pixel_disp << " px" << std::endl;

    // 视差过小则双帧初始化失败（相对图像宽度）
    const float image_width = Frame::mnMaxX - Frame::mnMinX;
    const float min_pixel_disp = kMinMeanPixelDispRatio * image_width;
    if(image_width <= 0.0f || mean_pixel_disp < min_pixel_disp)
    {
        if(mbVerbose)
            std::cout << "[TagInitializer] TryInitializeTwoFrames failed"
                      << " (insufficient parallax"
                      << ", mean_pixel_disp=" << mean_pixel_disp
                      << " px, need >= " << min_pixel_disp << " px)"
                      << std::endl;
        return false;
    }

    GeometricCamera *pCamera = frame.mpCamera;
    if(!pCamera)
        pCamera = ref_frame.mpCamera;
    if(!pCamera)
        return false;

    TwoViewAmbiguityResult ambiguity;
    if(!ResolveTwoViewIppeAmbiguity(common_obs, pCamera, ambiguity))
        return false;

    // Tag world := 参考帧相机系；当前帧位姿为 T_21
    Result out;
    out.Tcw_current = ambiguity.T_21;

    // 遍历消歧后的 Tag->Camera 位姿，写入初始化结果（统一用 Frame::mnId）
    for(const auto &kv : ambiguity.T_ct_ref)
    {
        const int tag_id = kv.first;
        const auto common_it = common_obs.find(tag_id);
        if(common_it == common_obs.end() || !common_it->second.ref ||
           !common_it->second.cur)
            continue;

        const auto cur_pose_it = ambiguity.T_ct_cur.find(tag_id);
        if(cur_pose_it == ambiguity.T_ct_cur.end())
            continue;

        // 获取消歧后指定Tag的参考帧/当前帧 IPPE 下标
        const auto sel_ref_it = ambiguity.selected_ref_candidate.find(tag_id);
        const auto sel_cur_it = ambiguity.selected_cur_candidate.find(tag_id);
        if(sel_ref_it == ambiguity.selected_ref_candidate.end() ||
           sel_cur_it == ambiguity.selected_cur_candidate.end())
            continue;

        const int i_ref = sel_ref_it->second;
        const int i_cur = sel_cur_it->second;
        const Sophus::SE3f &T_c1t = kv.second;
        const Sophus::SE3f &T_c2t = cur_pose_it->second;

        // 回写到 Frame.mTagFrameData：仅更新 selected_candidate 与所选候选解
        ref_frame.mTagFrameData.WriteBackSelectedCandidate(
            tag_id, common_it->second.ref->camera_id, i_ref, &T_c1t);
        frame.mTagFrameData.WriteBackSelectedCandidate(
            tag_id, common_it->second.cur->camera_id, i_cur, &T_c2t);

        // 从回写后的 Frame 取观测，注册到 MapTagData
        const TagObservation *obs_ref_ptr =
            ref_frame.mTagFrameData.Find(tag_id, common_it->second.ref->camera_id);
        const TagObservation *obs_cur_ptr =
            frame.mTagFrameData.Find(tag_id, common_it->second.cur->camera_id);
        if(!obs_ref_ptr || !obs_cur_ptr)
            continue;

        MapTagPtr map_tag = std::make_shared<MapTagData>();
        map_tag->tag_id = tag_id;
        map_tag->SetPose(T_c1t);  // T_wt = T_c1t（world = ref camera）
        map_tag->SetState(MapTagState::FIXED);
        map_tag->AddObservation(ref_frame.mnId, *obs_ref_ptr);
        map_tag->AddObservation(frame.mnId, *obs_cur_ptr);
        out.tags.emplace(tag_id, std::move(map_tag));
    }

    if(out.tags.empty())
        return false;

    // 供 TagKeyFrameDataBase 注册：参考帧 (Tcw=I) + 当前帧 (Tcw=T_21)
    // 此时 Frame.mTagFrameData 中对应 Tag 的 selected_candidate 已回写
    Frame kf_ref = ref_frame;
    kf_ref.SetTagPose(Sophus::SE3f());
    Frame kf_cur = frame;
    kf_cur.SetTagPose(out.Tcw_current);
    out.keyframes.push_back(std::move(kf_ref));
    out.keyframes.push_back(std::move(kf_cur));

    result = std::move(out);
    return true;
}

bool TagInitializer::TryInitialize(Frame &frame, Result &result)
{
    // 1) 始终优先单帧初始化
    if(TryInitializeSingleFrame(frame, result))
    {
        Clear();
        return true;
    }

    // 2) 单帧失败：尚无参考帧 → 将当前帧设为参考帧
    if(mInitFrames.empty())
    {
        SetReferenceFrame(frame);
        return false;
    }

    // 第二帧及之后
    Frame &ref_frame = mInitFrames.front();
    const CommonTagObsMap common_obs = FindCommonTagObservations(frame, ref_frame);

    if(mbVerbose)
        std::cout << "[TagInitializer] common tag observations: "
                  << common_obs.size()
                  << " (need >= " << kMinCommonTagsForTwoFrame << ")"
                  << std::endl;

    // 共同 Tag id 足够 → 尝试双帧初始化
    if(common_obs.size() >= kMinCommonTagsForTwoFrame)
    {
        if(TryInitializeTwoFrames(frame, ref_frame, common_obs, result))
        {
            Clear();
            return true;
        }
        // 双帧失败：保留参考帧不动
        return false;
    }

    // 共同 Tag 过少：切换参考帧为当前帧
    SetReferenceFrame(frame);
    return false;
}

}  // namespace tag
}  // namespace ORB_SLAM3
