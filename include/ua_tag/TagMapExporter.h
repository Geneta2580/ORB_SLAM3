#ifndef UA_TAG_TAG_MAP_EXPORTER_H
#define UA_TAG_TAG_MAP_EXPORTER_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Map;

namespace tag {

// 导出 Map 中 MapTag 世界系四角点与关联关键帧相机 Pose
class TagMapExporter
{
public:
    // output_dir 下写入 tag_map_corners.csv / tag_camera_trajectory.txt(TUM)
    explicit TagMapExporter(const std::string &output_dir);
    ~TagMapExporter();

    TagMapExporter(const TagMapExporter &) = delete;
    TagMapExporter &operator=(const TagMapExporter &) = delete;

    bool IsEnabled() const noexcept { return mbEnabled; }
    const std::string &OutputDir() const noexcept { return mOutputDir; }

    // 追加一帧相机轨迹（输入 Tcw；落盘 TUM：timestamp tx ty tz qx qy qz qw，位姿为 Twc）
    void AppendCameraPose(unsigned long frame_id,
                          double timestamp,
                          const Sophus::SE3f &Tcw);

    // 导出当前 Map 中全部 MapTag：tag_id + 四角点世界坐标
    bool SaveTagMapCorners(Map &map) const;

    // 导出有 MapTag 关联的关键帧位姿（Twc TUM）
    bool SaveTagInitKeyFrames(Map &map) const;

    // 导出定标后的 ORB 初始化地图点 + 两帧关键帧位姿（与 Tag 导出同目录，便于尺度对比）
    // orb_kf_tcw: (timestamp, Tcw)
    bool SaveOrbInitMap(
        const std::vector<Eigen::Vector3f> &points,
        const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw) const;

    // 关闭轨迹文件（SaveTagMapCorners 可重复调用）
    void CloseTrajectory();

private:
    bool OpenTrajectoryFile();

    std::string mOutputDir;
    bool mbEnabled{false};
    mutable std::ofstream mTrajFile;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
