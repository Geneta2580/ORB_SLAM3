#include "ua_tag/MapTagData.h"

namespace ORB_SLAM3 {
namespace tag {

int MapTagData::Id() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mnId;
}

void MapTagData::SetId(int id)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mnId = id;
}

float MapTagData::GetTagSize() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTagSize;
}

void MapTagData::SetTagSize(float tag_size)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTagSize = tag_size;
}

void MapTagData::SetPose(const Sophus::SE3f &T_wt)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTwt = T_wt;
    mbHasPose = true;
}

Sophus::SE3f MapTagData::GetPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwt;
}

bool MapTagData::HasPose() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasPose;
}

std::array<Eigen::Vector3f, 4> MapTagData::GetWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    std::array<Eigen::Vector3f, 4> corners{};
    if(!mbHasPose || mTagSize <= 0.0f)
        return corners;

    // 与 OpenCV IPPE_SQUARE / BuildSquareObjectPoints 约定一致
    const float h = mTagSize * 0.5f;
    const Eigen::Vector3f pts_t[4] = {
        Eigen::Vector3f(-h,  h, 0.f),
        Eigen::Vector3f( h,  h, 0.f),
        Eigen::Vector3f( h, -h, 0.f),
        Eigen::Vector3f(-h, -h, 0.f)};

    for(int i = 0; i < 4; ++i)
        corners[i] = mTwt * pts_t[i];
    return corners;
}

bool MapTagData::HasWorldCorners() const
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mbHasPose && mTagSize > 0.0f;
}

void MapTagData::SetState(MapTagState state)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mState = state;
}

MapTagState MapTagData::GetState() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState;
}

bool MapTagData::IsFixed() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState == MapTagState::FIXED_ANCHOR;
}

bool MapTagData::IsBad() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState == MapTagState::BAD;
}

Map *MapTagData::GetMap() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpMap;
}

void MapTagData::SetMapInternal(Map *pMap)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mpMap = pMap;
}

void MapTagData::AddObservationInternal(KeyFrame *pKF, int leftIndex, int rightIndex)
{
    if(!pKF)
        return;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    KeyFrameObservation obs;
    obs.leftIndex = leftIndex;
    obs.rightIndex = rightIndex;
    mObservations[pKF] = obs;
}

void MapTagData::EraseObservationInternal(KeyFrame *pKF)
{
    if(!pKF)
        return;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mObservations.erase(pKF);
}

bool MapTagData::IsInKeyFrame(KeyFrame *pKF) const
{
    if(!pKF)
        return false;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.find(pKF) != mObservations.end();
}

MapTagData::KeyFrameObservations MapTagData::GetObservations() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

std::size_t MapTagData::Observations() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.size();
}

}  // namespace tag
}  // namespace ORB_SLAM3
