#include "ua_tag/AprilTagVisualizer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

namespace ORB_SLAM3
{
namespace ua_tag
{

namespace
{

void DrawObservations(cv::Mat &vis, const tag::TagFrameData::Observations &tags)
{
    const cv::Scalar edgeColors[4] = {
        cv::Scalar(0, 255, 0),    // 0->1 green
        cv::Scalar(0, 255, 255),  // 1->2 yellow
        cv::Scalar(0, 128, 255),  // 2->3 orange
        cv::Scalar(255, 0, 0)     // 3->0 blue
    };

    for(size_t i = 0; i < tags.size(); ++i)
    {
        const tag::TagObservation &tag = tags[i];
        for(int k = 0; k < 4; ++k)
        {
            const cv::Point2f &p0 = tag.corners_raw[k];
            const cv::Point2f &p1 = tag.corners_raw[(k + 1) % 4];
            cv::line(vis, p0, p1, edgeColors[k], 2, cv::LINE_AA);
            cv::circle(vis, p0, 3, edgeColors[k], -1, cv::LINE_AA);
        }

        cv::Point2f center(0.f, 0.f);
        for(int k = 0; k < 4; ++k)
            center += tag.corners_raw[k];
        center *= 0.25f;

        // v: 位姿有效；a: 歧义已消解
        const bool has_pose = tag.pose_estimate.has_value();
        const tag::TagPoseCandidate *selected =
            has_pose ? tag.pose_estimate->Selected() : nullptr;
        const int valid = selected ? 1 : 0;
        const int amb_resolved =
            (has_pose && tag.pose_estimate->selected_candidate >= 0) ? 1 : 0;

        char line1[32];
        char line2[32];
        std::snprintf(line1, sizeof(line1), "id:%d", tag.tag_id);
        std::snprintf(line2, sizeof(line2), "v:%d a:%d", valid, amb_resolved);

        const cv::Scalar textColor = valid ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
        cv::putText(vis, line1, center + cv::Point2f(4.f, -8.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, textColor, 2, cv::LINE_AA);
        cv::putText(vis, line2, center + cv::Point2f(4.f, 10.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, textColor, 2, cv::LINE_AA);
    }
}

}  // namespace

bool EnsureDir(const std::string &dirPath)
{
    if(dirPath.empty())
        return false;

    struct stat st;
    if(stat(dirPath.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    return mkdir(dirPath.c_str(), 0755) == 0;
}

cv::Mat DrawTags(const cv::Mat &image, const tag::TagFrameData &frame_data)
{
    cv::Mat vis;
    if(image.empty())
        return vis;

    if(image.channels() == 1)
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    else if(image.channels() == 4)
        cv::cvtColor(image, vis, cv::COLOR_BGRA2BGR);
    else
        vis = image.clone();

    DrawObservations(vis, frame_data.left);
    DrawObservations(vis, frame_data.right);
    return vis;
}

bool SaveTagsVis(const cv::Mat &image,
                 const tag::TagFrameData &frame_data,
                 const std::string &savePath)
{
    if(savePath.empty() || frame_data.Empty())
        return false;

    cv::Mat vis = DrawTags(image, frame_data);
    if(vis.empty())
        return false;

    return cv::imwrite(savePath, vis);
}

}  // namespace ua_tag
}  // namespace ORB_SLAM3
