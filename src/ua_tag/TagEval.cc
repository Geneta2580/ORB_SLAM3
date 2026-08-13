#include "ua_tag/TagEval.h"

#include "CameraModels/GeometricCamera.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Tracking.h"
#include "ua_tag/TagFrameData.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

namespace {

bool EnsureDir(const std::string &dirPath)
{
    if(dirPath.empty())
        return false;
    struct stat st;
    if(stat(dirPath.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);
    return mkdir(dirPath.c_str(), 0755) == 0;
}

constexpr float kRad2Deg = 180.0f / static_cast<float>(M_PI);

float RotErrorDeg(const Sophus::SE3f &T)
{
    return T.so3().log().norm() * kRad2Deg;
}

std::string JoinSortedInts(std::vector<int> ids)
{
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::ostringstream oss;
    for(size_t i = 0; i < ids.size(); ++i)
    {
        if(i)
            oss << ';';
        oss << ids[i];
    }
    return oss.str();
}

std::string JoinTagIds(const std::vector<TagPoseConstraint> &tagSnap)
{
    std::vector<int> ids;
    ids.reserve(tagSnap.size());
    for(const TagPoseConstraint &c : tagSnap)
        ids.push_back(c.tagId);
    return JoinSortedInts(std::move(ids));
}

char CameraLetter(CameraId cam)
{
    return cam == CameraId::RIGHT ? 'R' : 'L';
}

void CollectPoseTagGroups(const Frame &frame,
                          const std::vector<TagPoseConstraint> &tagSnap,
                          std::vector<int> &outlier_ids,
                          std::vector<TagGroupOutlier> &groups)
{
    outlier_ids.clear();
    groups.clear();
    std::unordered_set<int> snap;
    snap.reserve(tagSnap.size() * 2 + 1);
    for(const TagPoseConstraint &c : tagSnap)
        snap.insert(c.tagId);

    auto add = [&](const TagObservation &o, CameraId cam) {
        if(!o.IsDetectValid() || snap.find(o.tag_id) == snap.end())
            return;
        TagGroupOutlier g;
        g.tag_id = o.tag_id;
        g.camera = cam;
        g.n_corners = 4;
        g.is_opt_outlier = o.is_opt_outlier;
        for(int k = 0; k < 4; ++k)
        {
            if(!o.corner_outliers[static_cast<size_t>(k)])
                ++g.n_inlier_corners;
        }
        groups.push_back(g);
        if(o.is_opt_outlier)
            outlier_ids.push_back(o.tag_id);
    };
    for(const TagObservation &o : frame.mTagFrameData.left)
        add(o, CameraId::LEFT_OR_MONO);
    for(const TagObservation &o : frame.mTagFrameData.right)
        add(o, CameraId::RIGHT);
}

Sophus::SE3f UmeyamaSE3(const std::vector<Eigen::Vector3f> &src,
                        const std::vector<Eigen::Vector3f> &dst)
{
    const int n = static_cast<int>(std::min(src.size(), dst.size()));
    Sophus::SE3f T;
    if(n < 1)
        return T;
    if(n == 1)
    {
        T.translation() = dst[0] - src[0];
        return T;
    }

    Eigen::Vector3d muSrc = Eigen::Vector3d::Zero();
    Eigen::Vector3d muDst = Eigen::Vector3d::Zero();
    for(int i = 0; i < n; ++i)
    {
        muSrc += src[i].cast<double>();
        muDst += dst[i].cast<double>();
    }
    muSrc /= n;
    muDst /= n;

    Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
    for(int i = 0; i < n; ++i)
    {
        const Eigen::Vector3d a = src[i].cast<double>() - muSrc;
        const Eigen::Vector3d b = dst[i].cast<double>() - muDst;
        sigma += b * a.transpose();
    }
    sigma /= n;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = U * V.transpose();
    if(R.determinant() < 0.0)
    {
        U.col(2) *= -1.0;
        R = U * V.transpose();
    }
    const Eigen::Vector3d t = muDst - R * muSrc;
    return Sophus::SE3f(R.cast<float>(), t.cast<float>());
}

}  // namespace

bool TagEval::MatToSE3(const cv::Mat &T, Sophus::SE3f &out)
{
    if(T.rows != 4 || T.cols != 4)
        return false;
    cv::Mat Td;
    T.convertTo(Td, CV_64F);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
            R(r, c) = static_cast<float>(Td.at<double>(r, c));
        t(r) = static_cast<float>(Td.at<double>(r, 3));
    }
    out = Sophus::SE3f(R, t);
    return true;
}

std::unique_ptr<TagEval> TagEval::FromSettings(const std::string &settings_path)
{
    auto eval = std::unique_ptr<TagEval>(new TagEval());
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened())
        return eval;

    cv::FileNode node = fs["Tag.eval.enable"];
    if(!node.empty())
        eval->mbEnable = static_cast<int>(node) != 0;
    if(!eval->mbEnable)
        return eval;

    node = fs["Tag.eval.out_dir"];
    if(!node.empty() && node.isString())
        eval->mOutDir = static_cast<std::string>(node);
    else
    {
        node = fs["Tag.export_dir"];
        if(!node.empty() && node.isString())
            eval->mOutDir = static_cast<std::string>(node);
    }

    node = fs["Tag.eval.gt_file"];
    if(!node.empty() && node.isString())
        eval->mGtFile = static_cast<std::string>(node);

    node = fs["Tag.eval.gt_frame"];
    if(!node.empty() && node.isString())
    {
        const std::string frame = static_cast<std::string>(node);
        if(frame == "imu" || frame == "body" || frame == "imu_body")
            eval->mGtFrame = GtFrame::Imu;
        else
            eval->mGtFrame = GtFrame::Camera;
    }

    node = fs["Tag.eval.gt_world_from_sensor"];
    if(!node.empty())
        eval->mbGtWorldFromSensor = static_cast<int>(node) != 0;

    node = fs["Tag.eval.gt_time_scale"];
    if(!node.empty())
        eval->mGtTimeScale = static_cast<double>(node);

    node = fs["Tag.eval.T_b_c"];
    if(node.empty())
        node = fs["IMU.T_b_c1"];
    if(!node.empty())
    {
        cv::Mat T;
        node >> T;
        if(MatToSE3(T, eval->mT_b_c))
            eval->mbHasTbc = true;
    }
    if(!eval->mbHasTbc)
    {
        cv::FileNode nq = fs["Tag.eval.T_b_c_q"];
        cv::FileNode nt = fs["Tag.eval.T_b_c_t"];
        if(!nq.empty() && nq.size() >= 4 && !nt.empty() && nt.size() >= 3)
        {
            const float qx = static_cast<float>(static_cast<double>(nq[0]));
            const float qy = static_cast<float>(static_cast<double>(nq[1]));
            const float qz = static_cast<float>(static_cast<double>(nq[2]));
            const float qw = static_cast<float>(static_cast<double>(nq[3]));
            Eigen::Quaternionf q(qw, qx, qy, qz);
            q.normalize();
            Eigen::Vector3f t(static_cast<float>(static_cast<double>(nt[0])),
                              static_cast<float>(static_cast<double>(nt[1])),
                              static_cast<float>(static_cast<double>(nt[2])));
            eval->mT_b_c = Sophus::SE3f(q, t);
            eval->mbHasTbc = true;
        }
    }

    node = fs["Tag.eval.align_mode"];
    if(!node.empty() && node.isString())
    {
        const std::string mode = static_cast<std::string>(node);
        if(mode == "none")
            eval->mAlignMode = AlignMode::None;
        else if(mode == "yaml")
            eval->mAlignMode = AlignMode::Yaml;
        else
            eval->mAlignMode = AlignMode::PreTag;
    }

    node = fs["Tag.eval.lba_every_n"];
    if(!node.empty())
        eval->mLbaEveryN = std::max(1, static_cast<int>(node));

    node = fs["Tag.eval.induced_trans_th"];
    if(!node.empty())
        eval->mInducedTransTh = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.eval.induced_rot_th_deg"];
    if(!node.empty())
        eval->mInducedRotThDeg = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.eval.delta_trans_th"];
    if(!node.empty())
        eval->mDeltaTransTh = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.eval.delta_rot_th_deg"];
    if(!node.empty())
        eval->mDeltaRotThDeg = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.eval.gt_max_dt"];
    if(!node.empty())
        eval->mGtMaxDt = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.eval.T_gt_slam"];
    if(!node.empty())
    {
        cv::Mat T;
        node >> T;
        if(MatToSE3(T, eval->mT_gt_slam) && eval->mAlignMode == AlignMode::Yaml)
            eval->mbAligned = true;
    }

    if(!eval->mGtFile.empty())
        eval->mbGtLoaded = eval->LoadGroundTruth(eval->mGtFile);

    if(eval->mGtFrame == GtFrame::Imu && !eval->mbHasTbc)
    {
        std::cerr << "[TagEval] gt_frame=imu but T_b_c / IMU.T_b_c1 is missing; "
                  << "IMU-camera lever arm will contaminate GT errors" << std::endl;
    }

    if(eval->mAlignMode == AlignMode::None)
    {
        eval->mT_gt_slam = Sophus::SE3f();
        eval->mbAligned = true;
    }

    if(!eval->OpenLogs())
        eval->mbEnable = false;
    else
    {
        std::cout << "[TagEval] enabled out_dir=" << eval->mOutDir
                  << " gt=" << (eval->mbGtLoaded ? eval->mGtFile : std::string("<none>"))
                  << " gt_frame=" << (eval->mGtFrame == GtFrame::Imu ? "imu" : "camera")
                  << " T_b_c=" << (eval->mbHasTbc ? 1 : 0)
                  << " align=" << (eval->mAlignMode == AlignMode::None ? "none"
                                     : (eval->mAlignMode == AlignMode::Yaml ? "yaml" : "pre_tag"))
                  << " lba_every_n=" << eval->mLbaEveryN << std::endl;
    }
    return eval;
}

bool TagEval::OpenLogs()
{
    if(!EnsureDir(mOutDir))
    {
        std::cerr << "[TagEval] failed to create dir: " << mOutDir << std::endl;
        return false;
    }

    const std::string pose_path = mOutDir + "/tag_eval_pose.csv";
    mPoseFile.open(pose_path.c_str(), std::ios::out | std::ios::trunc);
    const std::string pose_tag_path = mOutDir + "/tag_eval_pose_tag.csv";
    mPoseTagFile.open(pose_tag_path.c_str(), std::ios::out | std::ios::trunc);
    const std::string lba_path = mOutDir + "/tag_eval_lba.csv";
    mLbaFile.open(lba_path.c_str(), std::ios::out | std::ios::trunc);
    const std::string lba_kf_path = mOutDir + "/tag_eval_lba_kf.csv";
    mLbaKfFile.open(lba_kf_path.c_str(), std::ios::out | std::ios::trunc);
    const std::string lba_tag_path = mOutDir + "/tag_eval_lba_tag.csv";
    mLbaTagFile.open(lba_tag_path.c_str(), std::ios::out | std::ios::trunc);
    if(!mPoseFile.is_open() || !mPoseTagFile.is_open() || !mLbaFile.is_open()
       || !mLbaKfFile.is_open() || !mLbaTagFile.is_open())
    {
        std::cerr << "[TagEval] failed to open CSV logs in " << mOutDir << std::endl;
        return false;
    }

    mPoseFile << std::fixed
              << "timestamp,frame_id,tag_ids,outlier_tag_ids,orb_inliers,tag_corner_inliers,has_gt,aligned,"
              << "orb_gt_trans_error,fused_gt_trans_error,delta_gt_trans_error,"
              << "orb_gt_rot_error_deg,fused_gt_rot_error_deg,delta_gt_rot_error_deg,"
              << "tag_induced_translation,tag_induced_rotation_deg,"
              << "orb_rmse_at_orb_pose,orb_rmse_at_fused_pose,"
              << "tag_rmse_at_orb_pose,tag_rmse_at_fused_pose,"
              << "tag_acted,tag_pull_away\n";
    mPoseTagFile << std::fixed
                 << "timestamp,frame_id,tag_id,camera,n_corners,n_inlier_corners,is_opt_outlier\n";
    mLbaFile << std::fixed
             << "timestamp,current_kf_id,num_local_kfs,num_fixed_kfs,num_map_points,"
             << "num_tags,num_tag_edges,num_tag_outliers,outlier_tag_ids,n_kf_with_gt,"
             << "mean_delta_trans,p90_delta_trans,max_delta_trans,"
             << "mean_delta_rot_deg,p90_delta_rot_deg,max_delta_rot_deg,"
             << "n_harmed_kfs,n_helped_kfs,affected_frame_count,n_frame_with_gt,"
             << "mean_delta_trans_fr,p90_delta_trans_fr,max_delta_trans_fr,"
             << "mean_delta_rot_fr,p90_delta_rot_fr,max_delta_rot_fr,"
             << "n_harmed_frames,n_helped_frames,forced_common_gauge\n";
    mLbaKfFile << std::fixed
               << "timestamp,current_kf_id,kf_id,kf_timestamp,has_gt,"
               << "orb_gt_trans_error,fused_gt_trans_error,delta_gt_trans_error,"
               << "orb_gt_rot_error_deg,fused_gt_rot_error_deg,delta_gt_rot_error_deg,"
               << "tag_induced_translation,tag_induced_rotation_deg\n";
    mLbaTagFile << std::fixed
                << "timestamp,current_kf_id,kf_id,tag_id,camera,n_corners,n_inlier_corners,is_opt_outlier\n";
    return true;
}

bool TagEval::LoadGroundTruth(const std::string &path)
{
    std::ifstream f(path.c_str());
    if(!f.is_open())
    {
        std::cerr << "[TagEval] failed to open GT file: " << path << std::endl;
        return false;
    }

    mvGtTime.clear();
    mvGtTwc.clear();
    std::string line;
    while(std::getline(f, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        double t, px, py, pz, qx, qy, qz, qw;
        if(!(ss >> t >> px >> py >> pz >> qx >> qy >> qz >> qw))
            continue;
        t *= mGtTimeScale;
        Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx),
                             static_cast<float>(qy), static_cast<float>(qz));
        q.normalize();
        Sophus::SE3f T_ws(q, Eigen::Vector3f(static_cast<float>(px),
                                             static_cast<float>(py),
                                             static_cast<float>(pz)));
        if(!mbGtWorldFromSensor)
            T_ws = T_ws.inverse();
        // imu 文件是 Twb=vicon_T_imu；左相机 Twc = Twb * T_b_c
        if(mGtFrame == GtFrame::Imu && mbHasTbc)
            T_ws = T_ws * mT_b_c;
        mvGtTime.push_back(t);
        mvGtTwc.push_back(T_ws);
    }
    std::cout << "[TagEval] loaded " << mvGtTime.size() << " GT poses from " << path
              << " as left-camera Twc"
              << (mGtFrame == GtFrame::Imu ? " (from imu via T_b_c)" : "")
              << std::endl;
    return !mvGtTime.empty();
}

bool TagEval::InterpolateGtTwc(double timestamp, Sophus::SE3f &Twc_gt) const
{
    if(!mbGtLoaded || mvGtTime.empty())
        return false;
    if(timestamp < mvGtTime.front() - mGtMaxDt || timestamp > mvGtTime.back() + mGtMaxDt)
        return false;

    auto it = std::lower_bound(mvGtTime.begin(), mvGtTime.end(), timestamp);
    if(it == mvGtTime.begin())
    {
        if(std::abs(mvGtTime.front() - timestamp) > mGtMaxDt)
            return false;
        Twc_gt = mvGtTwc.front();
        return true;
    }
    if(it == mvGtTime.end())
    {
        if(std::abs(mvGtTime.back() - timestamp) > mGtMaxDt)
            return false;
        Twc_gt = mvGtTwc.back();
        return true;
    }

    const size_t i1 = static_cast<size_t>(std::distance(mvGtTime.begin(), it));
    const size_t i0 = i1 - 1;
    const double t0 = mvGtTime[i0];
    const double t1 = mvGtTime[i1];
    if(timestamp - t0 > mGtMaxDt && t1 - timestamp > mGtMaxDt)
        return false;
    const double dt = t1 - t0;
    const float a = (dt <= 1e-12) ? 0.0f : static_cast<float>((timestamp - t0) / dt);
    const Eigen::Vector3f p = (1.0f - a) * mvGtTwc[i0].translation() + a * mvGtTwc[i1].translation();
    const Eigen::Quaternionf q = mvGtTwc[i0].unit_quaternion().slerp(a, mvGtTwc[i1].unit_quaternion());
    Twc_gt = Sophus::SE3f(q, p);
    return true;
}

Sophus::SE3f TagEval::AlignEstTwc(const Sophus::SE3f &Twc_est) const
{
    if(!mbAligned)
        return Twc_est;
    return mT_gt_slam * Twc_est;
}

bool TagEval::ComputeGtErrors(const Sophus::SE3f &Tcw_est, double timestamp,
                              float &trans_m, float &rot_deg) const
{
    Sophus::SE3f Twc_gt;
    if(!InterpolateGtTwc(timestamp, Twc_gt))
        return false;
    const Sophus::SE3f Twc_est = AlignEstTwc(Tcw_est.inverse());
    const Sophus::SE3f E = Twc_gt.inverse() * Twc_est;
    trans_m = E.translation().norm();
    rot_deg = RotErrorDeg(E);
    return true;
}

void TagEval::TryFreezePreTagAlignment()
{
    if(mAlignMode != AlignMode::PreTag || mbAligned)
        return;
    if(!mbGtLoaded || mvAlignTwc.size() < 3)
    {
        if(!mvAlignTwc.empty() && mbGtLoaded)
        {
            Sophus::SE3f Twc_gt;
            if(InterpolateGtTwc(mvAlignTime.back(), Twc_gt))
            {
                mT_gt_slam = Twc_gt * mvAlignTwc.back().inverse();
                mbAligned = true;
                std::cout << "[TagEval] pre_tag alignment from 1 pose (buffer<"
                          << mvAlignTwc.size() << ")" << std::endl;
            }
        }
        mvAlignTime.clear();
        mvAlignTwc.clear();
        return;
    }

    std::vector<Eigen::Vector3f> src, dst;
    src.reserve(mvAlignTwc.size());
    dst.reserve(mvAlignTwc.size());
    for(size_t i = 0; i < mvAlignTwc.size(); ++i)
    {
        Sophus::SE3f Twc_gt;
        if(!InterpolateGtTwc(mvAlignTime[i], Twc_gt))
            continue;
        src.push_back(mvAlignTwc[i].translation());
        dst.push_back(Twc_gt.translation());
    }
    if(src.size() < 3)
    {
        std::cerr << "[TagEval] pre_tag alignment: not enough GT-matched poses ("
                  << src.size() << ")" << std::endl;
        mvAlignTime.clear();
        mvAlignTwc.clear();
        return;
    }
    mT_gt_slam = UmeyamaSE3(src, dst);
    mbAligned = true;
    std::cout << "[TagEval] pre_tag SE3 alignment from " << src.size()
              << " poses, t_norm=" << mT_gt_slam.translation().norm()
              << " rot_deg=" << RotErrorDeg(mT_gt_slam) << std::endl;
    mvAlignTime.clear();
    mvAlignTwc.clear();
}

void TagEval::ObservePoseForAlignment(double timestamp, const Sophus::SE3f &Tcw,
                                      bool tag_map_ready)
{
    if(!mbEnable || mAlignMode != AlignMode::PreTag)
        return;
    std::lock_guard<std::mutex> lock(mMutex);
    if(mbAligned)
        return;
    if(!tag_map_ready)
    {
        mvAlignTime.push_back(timestamp);
        mvAlignTwc.push_back(Tcw.inverse());
        return;
    }
    TryFreezePreTagAlignment();
}

bool TagEval::WantLbaShadow()
{
    if(!mbEnable)
        return false;
    std::lock_guard<std::mutex> lock(mMutex);
    ++mnLbaTagWindows;
    return (mnLbaTagWindows % mLbaEveryN) == 0;
}

void TagEval::SaveFramePoseState(const Frame &frame, FramePoseOptState &state)
{
    state.has_pose = frame.HasPose();
    if(state.has_pose)
        state.Tcw = frame.GetPose();
    state.mvbOutlier = frame.mvbOutlier;
    auto copy_flags = [](const TagFrameData::Observations &src,
                         std::vector<FramePoseOptState::TagFlags> &dst) {
        dst.resize(src.size());
        for(size_t i = 0; i < src.size(); ++i)
        {
            dst[i].is_opt_outlier = src[i].is_opt_outlier;
            dst[i].corner_outliers = src[i].corner_outliers;
        }
    };
    copy_flags(frame.mTagFrameData.left, state.left);
    copy_flags(frame.mTagFrameData.right, state.right);
}

void TagEval::RestoreFramePoseState(Frame &frame, const FramePoseOptState &state)
{
    if(state.has_pose)
        frame.SetPose(state.Tcw);
    if(state.mvbOutlier.size() == frame.mvbOutlier.size())
        frame.mvbOutlier = state.mvbOutlier;
    auto restore_flags = [](TagFrameData::Observations &dst,
                            const std::vector<FramePoseOptState::TagFlags> &src) {
        const size_t n = std::min(dst.size(), src.size());
        for(size_t i = 0; i < n; ++i)
        {
            dst[i].is_opt_outlier = src[i].is_opt_outlier;
            dst[i].corner_outliers = src[i].corner_outliers;
        }
    };
    restore_flags(frame.mTagFrameData.left, state.left);
    restore_flags(frame.mTagFrameData.right, state.right);
}

int TagEval::CountTagCornerInliers(const Frame &frame)
{
    int n = 0;
    auto count_obs = [&n](const TagFrameData::Observations &obs) {
        for(const TagObservation &o : obs)
        {
            if(!o.IsDetectValid() || o.is_opt_outlier)
                continue;
            for(int k = 0; k < 4; ++k)
            {
                if(!o.corner_outliers[static_cast<size_t>(k)])
                    ++n;
            }
        }
    };
    count_obs(frame.mTagFrameData.left);
    count_obs(frame.mTagFrameData.right);
    return n;
}

float TagEval::ComputeOrbRmse(const Frame &frame)
{
    if(!frame.HasPose() || !frame.mpCamera)
        return -1.0f;
    const Sophus::SE3f Tcw = frame.GetPose();
    double sum = 0.0;
    int n = 0;
    const int N = frame.N;
    for(int i = 0; i < N; ++i)
    {
        MapPoint *pMP = frame.mvpMapPoints[i];
        if(!pMP || (i < static_cast<int>(frame.mvbOutlier.size()) && frame.mvbOutlier[i]))
            continue;
        const Eigen::Vector3f Pc = Tcw * pMP->GetWorldPos();
        if(Pc(2) <= 1e-6f)
            continue;

        cv::Point2f uv_obs;
        GeometricCamera *cam = frame.mpCamera;
        if(frame.Nleft == -1)
        {
            uv_obs = frame.mvKeysUn[i].pt;
        }
        else if(i < frame.Nleft)
        {
            uv_obs = frame.mvKeys[i].pt;
        }
        else
        {
            if(!frame.mpCamera2)
                continue;
            cam = frame.mpCamera2;
            const Eigen::Vector3f PcR = frame.GetRelativePoseTrl() * Pc;
            if(PcR(2) <= 1e-6f)
                continue;
            const Eigen::Vector2f uv = cam->project(PcR);
            uv_obs = frame.mvKeysRight[i - frame.Nleft].pt;
            const float dx = uv(0) - uv_obs.x;
            const float dy = uv(1) - uv_obs.y;
            sum += dx * dx + dy * dy;
            ++n;
            continue;
        }
        const Eigen::Vector2f uv = cam->project(Pc);
        const float dx = uv(0) - uv_obs.x;
        const float dy = uv(1) - uv_obs.y;
        sum += dx * dx + dy * dy;
        ++n;
    }
    if(n == 0)
        return -1.0f;
    return static_cast<float>(std::sqrt(sum / n));
}

float TagEval::ComputeTagRmse(const Frame &frame,
                              const std::vector<TagPoseConstraint> &tagSnap,
                              bool use_opt_inliers)
{
    if(!frame.HasPose() || !frame.mpCamera || tagSnap.empty())
        return -1.0f;

    std::unordered_map<int, const TagPoseConstraint *> byId;
    byId.reserve(tagSnap.size());
    for(const TagPoseConstraint &c : tagSnap)
        byId[c.tagId] = &c;

    const Sophus::SE3f Tcw = frame.GetPose();
    double sum = 0.0;
    int n = 0;

    auto add_obs = [&](const TagObservation &obs, bool is_right) {
        if(!obs.IsDetectValid())
            return;
        if(use_opt_inliers && obs.is_opt_outlier)
            return;
        const auto it = byId.find(obs.tag_id);
        if(it == byId.end())
            return;
        GeometricCamera *cam = is_right ? frame.mpCamera2 : frame.mpCamera;
        if(!cam)
            return;
        const bool fisheye = cam->GetType() == GeometricCamera::CAM_FISHEYE;
        const auto &uv = fisheye ? obs.corners_raw : obs.corners_undistorted;
        const Sophus::SE3f T_cam_w = is_right ? (frame.GetRelativePoseTrl() * Tcw) : Tcw;
        for(int k = 0; k < 4; ++k)
        {
            if(use_opt_inliers && obs.corner_outliers[static_cast<size_t>(k)])
                continue;
            const Eigen::Vector3f Pw = it->second->worldCorners[static_cast<size_t>(k)].cast<float>();
            const Eigen::Vector3f Pc = T_cam_w * Pw;
            if(Pc(2) <= 1e-6f)
                continue;
            const Eigen::Vector2f proj = cam->project(Pc);
            const float dx = proj(0) - uv[static_cast<size_t>(k)].x;
            const float dy = proj(1) - uv[static_cast<size_t>(k)].y;
            sum += dx * dx + dy * dy;
            ++n;
        }
    };

    for(const TagObservation &obs : frame.mTagFrameData.left)
        add_obs(obs, false);
    for(const TagObservation &obs : frame.mTagFrameData.right)
        add_obs(obs, true);

    if(n == 0)
        return -1.0f;
    return static_cast<float>(std::sqrt(sum / n));
}

void TagEval::LogPoseShadow(const Frame &frame,
                            const std::vector<TagPoseConstraint> &tagSnap,
                            const Sophus::SE3f &Tcw_orb,
                            const Sophus::SE3f &Tcw_fused,
                            int orb_inliers,
                            int tag_corner_inliers,
                            float orb_rmse_at_orb,
                            float orb_rmse_at_fused,
                            float tag_rmse_at_orb,
                            float tag_rmse_at_fused)
{
    if(!mbEnable || !mPoseFile.is_open())
        return;

    const Sophus::SE3f dT = Tcw_fused * Tcw_orb.inverse();
    const float induced_t = dT.translation().norm();
    const float induced_r = RotErrorDeg(dT);

    float orb_t = -1.0f, fused_t = -1.0f, orb_r = -1.0f, fused_r = -1.0f;
    bool has_gt = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        has_gt = ComputeGtErrors(Tcw_orb, frame.mTimeStamp, orb_t, orb_r)
              && ComputeGtErrors(Tcw_fused, frame.mTimeStamp, fused_t, fused_r);
    }
    const float delta_t = has_gt ? (fused_t - orb_t) : 0.0f;
    const float delta_r = has_gt ? (fused_r - orb_r) : 0.0f;
    const int tag_acted = (induced_t > mInducedTransTh || induced_r > mInducedRotThDeg) ? 1 : 0;
    const int pull_away = (tag_acted && has_gt &&
                           (delta_t > mDeltaTransTh || delta_r > mDeltaRotThDeg)) ? 1 : 0;

    std::vector<int> outlier_ids;
    std::vector<TagGroupOutlier> groups;
    CollectPoseTagGroups(frame, tagSnap, outlier_ids, groups);

    std::lock_guard<std::mutex> lock(mMutex);
    mPoseFile << std::setprecision(6) << frame.mTimeStamp << ","
              << frame.mnId << ","
              << JoinTagIds(tagSnap) << ","
              << JoinSortedInts(outlier_ids) << ","
              << orb_inliers << ","
              << tag_corner_inliers << ","
              << (has_gt ? 1 : 0) << ","
              << (mbAligned ? 1 : 0) << ","
              << std::setprecision(6)
              << orb_t << "," << fused_t << "," << delta_t << ","
              << orb_r << "," << fused_r << "," << delta_r << ","
              << induced_t << "," << induced_r << ","
              << orb_rmse_at_orb << "," << orb_rmse_at_fused << ","
              << tag_rmse_at_orb << "," << tag_rmse_at_fused << ","
              << tag_acted << "," << pull_away << "\n";
    mPoseFile.flush();

    if(mPoseTagFile.is_open())
    {
        for(const TagGroupOutlier &g : groups)
        {
            mPoseTagFile << std::setprecision(6) << frame.mTimeStamp << ","
                         << frame.mnId << ","
                         << g.tag_id << ","
                         << CameraLetter(g.camera) << ","
                         << g.n_corners << ","
                         << g.n_inlier_corners << ","
                         << (g.is_opt_outlier ? 1 : 0) << "\n";
        }
        mPoseTagFile.flush();
    }
}

float TagEval::Percentile(std::vector<float> values, float q)
{
    if(values.empty())
        return 0.0f;
    std::sort(values.begin(), values.end());
    const float idx = q * static_cast<float>(values.size() - 1);
    const size_t i0 = static_cast<size_t>(idx);
    const size_t i1 = std::min(i0 + 1, values.size() - 1);
    const float a = idx - static_cast<float>(i0);
    return (1.0f - a) * values[i0] + a * values[i1];
}

Sophus::SE3f TagEval::ReconstructTcw(const Sophus::SE3f &Tcr, KeyFrame *pKF,
                                     const std::map<unsigned long, Sophus::SE3f> &override_poses)
{
    Sophus::SE3f Trw;
    while(pKF && pKF->isBad())
    {
        Trw = Trw * pKF->mTcp;
        pKF = pKF->GetParent();
    }
    if(!pKF)
        return Sophus::SE3f();
    const auto it = override_poses.find(pKF->mnId);
    const Sophus::SE3f Tcw_ref = (it != override_poses.end()) ? it->second : pKF->GetPose();
    return Tcr * Trw * Tcw_ref;
}

void TagEval::LogLbaShadow(KeyFrame *pCurKF,
                           const LocalBAShadowResult &orb,
                           const LocalBAShadowResult &fused,
                           Tracking *pTracker)
{
    if(!mbEnable || !mLbaFile.is_open() || !pCurKF)
        return;

    std::vector<float> dtrans, drot, dtrans_fr, drot_fr;
    int n_kf_gt = 0, n_harmed_kf = 0, n_helped_kf = 0;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        for(KeyFrame *pKFi : fused.local_kfs)
        {
            if(!pKFi)
                continue;
            const auto itF = fused.local_kf_poses.find(pKFi->mnId);
            const auto itO = orb.local_kf_poses.find(pKFi->mnId);
            if(itF == fused.local_kf_poses.end() || itO == orb.local_kf_poses.end())
                continue;
            const Sophus::SE3f &Tcw_o = itO->second;
            const Sophus::SE3f &Tcw_f = itF->second;
            const Sophus::SE3f dT = Tcw_f * Tcw_o.inverse();
            const float induced_t = dT.translation().norm();
            const float induced_r = RotErrorDeg(dT);

            float orb_t = -1.0f, fused_t = -1.0f, orb_r = -1.0f, fused_r = -1.0f;
            const bool has_gt = ComputeGtErrors(Tcw_o, pKFi->mTimeStamp, orb_t, orb_r)
                             && ComputeGtErrors(Tcw_f, pKFi->mTimeStamp, fused_t, fused_r);
            const float dt = has_gt ? (fused_t - orb_t) : 0.0f;
            const float dr = has_gt ? (fused_r - orb_r) : 0.0f;
            if(has_gt)
            {
                ++n_kf_gt;
                dtrans.push_back(dt);
                drot.push_back(dr);
                if(dt > mDeltaTransTh || dr > mDeltaRotThDeg)
                    ++n_harmed_kf;
                else if(dt < -mDeltaTransTh || dr < -mDeltaRotThDeg)
                    ++n_helped_kf;
            }
            if(mLbaKfFile.is_open())
            {
                mLbaKfFile << std::setprecision(6) << pCurKF->mTimeStamp << ","
                           << pCurKF->mnId << "," << pKFi->mnId << ","
                           << pKFi->mTimeStamp << ","
                           << (has_gt ? 1 : 0) << ","
                           << orb_t << "," << fused_t << "," << dt << ","
                           << orb_r << "," << fused_r << "," << dr << ","
                           << induced_t << "," << induced_r << "\n";
            }
        }
        if(mLbaKfFile.is_open())
            mLbaKfFile.flush();
    }

    int affected = 0, n_fr_gt = 0, n_harmed_fr = 0, n_helped_fr = 0;
    if(pTracker)
    {
        std::vector<Sophus::SE3f> Tcr;
        std::vector<KeyFrame*> refs;
        std::vector<double> times;
        std::vector<bool> lost;
        pTracker->CopyTrajectorySnapshot(Tcr, refs, times, lost);
        const size_t n = std::min(std::min(Tcr.size(), refs.size()),
                                  std::min(times.size(), lost.size()));
        std::lock_guard<std::mutex> lock(mMutex);
        for(size_t i = 0; i < n; ++i)
        {
            if(lost[i])
                continue;
            KeyFrame *pRef = refs[i];
            KeyFrame *pWalk = pRef;
            while(pWalk && pWalk->isBad())
                pWalk = pWalk->GetParent();
            if(!pWalk || fused.local_kf_poses.find(pWalk->mnId) == fused.local_kf_poses.end())
                continue;
            ++affected;
            const Sophus::SE3f Tcw_o = ReconstructTcw(Tcr[i], pRef, orb.local_kf_poses);
            const Sophus::SE3f Tcw_f = ReconstructTcw(Tcr[i], pRef, fused.local_kf_poses);
            float orb_t = -1.0f, fused_t = -1.0f, orb_r = -1.0f, fused_r = -1.0f;
            if(!ComputeGtErrors(Tcw_o, times[i], orb_t, orb_r)
               || !ComputeGtErrors(Tcw_f, times[i], fused_t, fused_r))
                continue;
            ++n_fr_gt;
            const float dt = fused_t - orb_t;
            const float dr = fused_r - orb_r;
            dtrans_fr.push_back(dt);
            drot_fr.push_back(dr);
            if(dt > mDeltaTransTh || dr > mDeltaRotThDeg)
                ++n_harmed_fr;
            else if(dt < -mDeltaTransTh || dr < -mDeltaRotThDeg)
                ++n_helped_fr;
        }
    }

    auto mean_of = [](const std::vector<float> &v) -> float {
        if(v.empty())
            return 0.0f;
        double s = 0.0;
        for(float x : v)
            s += x;
        return static_cast<float>(s / v.size());
    };
    auto max_of = [](const std::vector<float> &v) -> float {
        if(v.empty())
            return 0.0f;
        return *std::max_element(v.begin(), v.end());
    };

    std::lock_guard<std::mutex> lock(mMutex);
    mLbaFile << std::setprecision(6) << pCurKF->mTimeStamp << ","
             << pCurKF->mnId << ","
             << fused.num_opt_kf << ","
             << fused.num_fixed_kf << ","
             << fused.num_mps << ","
             << fused.num_tags << ","
             << fused.num_tag_edges << ","
             << fused.num_tag_outliers << ","
             << JoinSortedInts(fused.outlier_tag_ids) << ","
             << n_kf_gt << ","
             << mean_of(dtrans) << "," << Percentile(dtrans, 0.9f) << "," << max_of(dtrans) << ","
             << mean_of(drot) << "," << Percentile(drot, 0.9f) << "," << max_of(drot) << ","
             << n_harmed_kf << "," << n_helped_kf << ","
             << affected << "," << n_fr_gt << ","
             << mean_of(dtrans_fr) << "," << Percentile(dtrans_fr, 0.9f) << "," << max_of(dtrans_fr) << ","
             << mean_of(drot_fr) << "," << Percentile(drot_fr, 0.9f) << "," << max_of(drot_fr) << ","
             << n_harmed_fr << "," << n_helped_fr << ","
             << (fused.forced_common_gauge ? 1 : 0) << "\n";
    mLbaFile.flush();
    WriteLbaTagGroups(pCurKF, fused.tag_groups);
}

void TagEval::WriteLbaTagGroups(KeyFrame *pCurKF, const std::vector<TagGroupOutlier> &groups)
{
    if(!mLbaTagFile.is_open() || !pCurKF || groups.empty())
        return;
    for(const TagGroupOutlier &g : groups)
    {
        mLbaTagFile << std::setprecision(6) << pCurKF->mTimeStamp << ","
                    << pCurKF->mnId << ","
                    << g.kf_id << ","
                    << g.tag_id << ","
                    << CameraLetter(g.camera) << ","
                    << g.n_corners << ","
                    << g.n_inlier_corners << ","
                    << (g.is_opt_outlier ? 1 : 0) << "\n";
    }
    mLbaTagFile.flush();
}

void TagEval::LogLbaTagGroups(KeyFrame *pCurKF, const std::vector<TagGroupOutlier> &groups)
{
    if(!mbEnable || !pCurKF || groups.empty())
        return;
    std::lock_guard<std::mutex> lock(mMutex);
    WriteLbaTagGroups(pCurKF, groups);
}

}  // namespace tag
}  // namespace ORB_SLAM3
