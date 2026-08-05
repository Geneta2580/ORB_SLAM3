#include "ua_tag/TagMapExporter.h"

#include "KeyFrame.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

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

}  // namespace

TagMapExporter::TagMapExporter(const std::string &output_dir)
    : mOutputDir(output_dir)
{
    if(mOutputDir.empty())
        return;

    if(!EnsureDir(mOutputDir))
    {
        std::cerr << "[TagMapExporter] failed to create dir: " << mOutputDir
                  << std::endl;
        return;
    }

    mbEnabled = OpenTrajectoryFile();
    if(mbEnabled)
        std::cout << "[TagMapExporter] export dir=" << mOutputDir << std::endl;
}

TagMapExporter::~TagMapExporter()
{
    CloseTrajectory();
}

bool TagMapExporter::OpenTrajectoryFile()
{
    // TUM 轨迹：timestamp tx ty tz qx qy qz qw（Twc）
    const std::string path = mOutputDir + "/tag_camera_trajectory.txt";
    mTrajFile.open(path.c_str(), std::ios::out | std::ios::trunc);
    if(!mTrajFile.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    mTrajFile << std::fixed;
    return true;
}

void TagMapExporter::AppendCameraPose(unsigned long frame_id,
                                      double timestamp,
                                      const Sophus::SE3f &Tcw)
{
    (void)frame_id;
    if(!mbEnabled || !mTrajFile.is_open())
        return;

    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Quaternionf q = Twc.unit_quaternion();
    const Eigen::Vector3f t = Twc.translation();

    mTrajFile << std::setprecision(6) << timestamp << " "
              << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}

bool TagMapExporter::SaveTagMapCorners(Map &map, bool verbose) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/tag_map_corners.csv";
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }

    ofs << std::setprecision(17);
    ofs << "tag_id,is_fixed,has_corners,"
        << "c0_x,c0_y,c0_z,"
        << "c1_x,c1_y,c1_z,"
        << "c2_x,c2_y,c2_z,"
        << "c3_x,c3_y,c3_z\n";

    std::vector<Map::MapTagPtr> tags = map.GetAllMapTags();
    // 按 tag_id 排序，便于对比
    std::sort(tags.begin(), tags.end(),
              [](const Map::MapTagPtr &a, const Map::MapTagPtr &b) {
                  if(!a)
                      return true;
                  if(!b)
                      return false;
                  return a->Id() < b->Id();
              });

    std::size_t n_with_corners = 0;
    for(const auto &map_tag : tags)
    {
        if(!map_tag)
            continue;

        ofs << map_tag->Id() << ","
            << (map_tag->IsFixed() ? 1 : 0) << ","
            << (map_tag->HasWorldCorners() ? 1 : 0);

        if(map_tag->HasWorldCorners())
        {
            const auto c = map_tag->GetWorldCorners();
            for(int i = 0; i < 4; ++i)
                ofs << "," << c[i].x() << "," << c[i].y() << "," << c[i].z();
            ++n_with_corners;
        }
        else
        {
            ofs << ",,,,,,,,,,,,";
        }
        ofs << "\n";
    }

    ofs.close();
    if(verbose)
    {
        std::cout << "[TagMapExporter] saved tag map corners -> " << path
                  << " num_tags=" << tags.size()
                  << " with_corners=" << n_with_corners << std::endl;
    }
    return true;
}

bool TagMapExporter::SaveTagInitKeyFrames(Map &map) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/tag_init_keyframes_tum.txt";
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }

    ofs << std::fixed;
    std::vector<KeyFrame *> kfs = map.GetAllKeyFrames();
    std::sort(kfs.begin(), kfs.end(),
              [](KeyFrame *a, KeyFrame *b) {
                  if(!a)
                      return true;
                  if(!b)
                      return false;
                  return a->mTimeStamp < b->mTimeStamp;
              });

    std::size_t n = 0;
    for(KeyFrame *pKF : kfs)
    {
        if(!pKF || pKF->isBad())
            continue;
        if(pKF->GetMapTagMatches().empty())
            continue;

        const Sophus::SE3f Twc = pKF->GetPoseInverse();
        const Eigen::Quaternionf q = Twc.unit_quaternion();
        const Eigen::Vector3f t = Twc.translation();
        ofs << std::setprecision(6) << pKF->mTimeStamp << " "
            << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        ++n;
    }

    ofs.close();
    std::cout << "[TagMapExporter] saved tag init keyframes -> " << path
              << " num_kf=" << n << std::endl;
    return true;
}

bool TagMapExporter::SaveOrbInitMap(
    const std::vector<Eigen::Vector3f> &points,
    const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string pts_path = mOutputDir + "/orb_init_map_points.csv";
    {
        std::ofstream ofs(pts_path.c_str(), std::ios::out | std::ios::trunc);
        if(!ofs.is_open())
        {
            std::cerr << "[TagMapExporter] failed to open: " << pts_path << std::endl;
            return false;
        }
        ofs << std::setprecision(17);
        ofs << "x,y,z\n";
        for(const Eigen::Vector3f &p : points)
            ofs << p.x() << "," << p.y() << "," << p.z() << "\n";
        ofs.close();
        std::cout << "[TagMapExporter] saved ORB init map points -> " << pts_path
                  << " num_pts=" << points.size() << std::endl;
    }

    const std::string kf_path = mOutputDir + "/orb_init_keyframes_tum.txt";
    {
        std::ofstream ofs(kf_path.c_str(), std::ios::out | std::ios::trunc);
        if(!ofs.is_open())
        {
            std::cerr << "[TagMapExporter] failed to open: " << kf_path << std::endl;
            return false;
        }
        ofs << std::fixed;
        for(const auto &item : orb_kf_tcw)
        {
            const Sophus::SE3f Twc = item.second.inverse();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            const Eigen::Vector3f t = Twc.translation();
            ofs << std::setprecision(6) << item.first << " "
                << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z()
                << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                << "\n";
        }
        ofs.close();
        std::cout << "[TagMapExporter] saved ORB init keyframes -> " << kf_path
                  << " num_kf=" << orb_kf_tcw.size() << std::endl;
    }

    return true;
}

void TagMapExporter::CloseTrajectory()
{
    if(mTrajFile.is_open())
    {
        mTrajFile.flush();
        mTrajFile.close();
    }
}

}  // namespace tag
}  // namespace ORB_SLAM3
