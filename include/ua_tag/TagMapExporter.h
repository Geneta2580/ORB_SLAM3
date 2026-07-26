#ifndef UA_TAG_TAG_MAP_EXPORTER_H
#define UA_TAG_TAG_MAP_EXPORTER_H

#include <fstream>
#include <string>

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {
namespace tag {

class TagMap;

// 导出 TagMap 世界系四角点与相机 TagPose 轨迹
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

    // 导出当前 TagMap：tag_id + 四角点世界坐标
    bool SaveTagMapCorners(const TagMap &tag_map) const;

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
