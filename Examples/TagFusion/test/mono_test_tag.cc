/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<sstream>
#include<dirent.h>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

bool ParseTimestampFromFilename(const string &filename, double &timestamp);
void LoadImages(const string &strImagePath,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_test_tag path_to_vocabulary path_to_settings path_to_image_folder (trajectory_file_name)" << endl;
        cerr << "Image filenames must contain timestamps, e.g. frame_1607984240.523067951.png" << endl;
        return 1;
    }

    bool bFileName = (argc >= 5);
    if (bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;

    cout << "Loading images from " << argv[3] << "...";
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestampsCam);
    cout << "LOADED " << vstrImageFilenames.size() << " images!" << endl;

    if(vstrImageFilenames.empty())
    {
        cerr << "ERROR: No valid images found in " << argv[3] << endl;
        return 1;
    }

    vector<float> vTimesTrack(vstrImageFilenames.size());

    cout << endl << "-------" << endl;
    cout.precision(17);

    // 先建 System，复用其已加载的相机内参；Tag 角点检测在 TrackMonocular → Frame 内完成
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    cv::Mat im;
    for(size_t ni = 0; ni < vstrImageFilenames.size(); ni++)
    {
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = 0;
        if(ni + 1 < vTimestampsCam.size())
            T = vTimestampsCam[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();

    if (bFileName)
    {
        const string kf_file = "kf_" + string(argv[4]) + ".txt";
        const string f_file = "f_" + string(argv[4]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

bool ParseTimestampFromFilename(const string &filename, double &timestamp)
{
    const string prefix = "frame_";
    const string suffix = ".png";

    if(filename.size() <= prefix.size() + suffix.size())
        return false;
    if(filename.compare(0, prefix.size(), prefix) != 0)
        return false;
    if(filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    string tsStr = filename.substr(prefix.size(), filename.size() - prefix.size() - suffix.size());
    if(tsStr.empty())
        return false;

    try
    {
        timestamp = stod(tsStr);
    }
    catch(...)
    {
        return false;
    }

    return true;
}

void LoadImages(const string &strImagePath,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    vector<pair<double, string> > vAllImages;

    DIR *dir = opendir(strImagePath.c_str());
    if(!dir)
    {
        cerr << endl << "ERROR: Cannot open image directory: " << strImagePath << endl;
        return;
    }

    struct dirent *entry;
    while((entry = readdir(dir)) != NULL)
    {
        string name = entry->d_name;
        if(name == "." || name == "..")
            continue;

        double t;
        if(!ParseTimestampFromFilename(name, t))
            continue;

        vAllImages.push_back(make_pair(t, strImagePath + "/" + name));
    }
    closedir(dir);

    sort(vAllImages.begin(), vAllImages.end());

    vstrImages.reserve(vAllImages.size());
    vTimeStamps.reserve(vAllImages.size());
    for(size_t i = 0; i < vAllImages.size(); i++)
    {
        vTimeStamps.push_back(vAllImages[i].first);
        vstrImages.push_back(vAllImages[i].second);
    }
}
