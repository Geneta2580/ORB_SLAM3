#ifndef UA_TAG_APRILTAG_VISUALIZER_H
#define UA_TAG_APRILTAG_VISUALIZER_H

#include "ua_tag/TagFrameData.h"

#include <opencv2/core/core.hpp>
#include <string>

namespace ORB_SLAM3
{
namespace ua_tag
{

// 在图像上绘制 Tag：id / 位姿有效性(v) / 歧义是否解决(a)
cv::Mat DrawTags(const cv::Mat &image, const tag::TagFrameData &frame_data);

// 绘制并保存到本地；成功返回 true
bool SaveTagsVis(const cv::Mat &image,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath);

// 确保输出目录存在；成功返回 true
bool EnsureDir(const std::string &dirPath);

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif
