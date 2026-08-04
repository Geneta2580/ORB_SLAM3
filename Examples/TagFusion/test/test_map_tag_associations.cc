/**
 * MapTag 数据模型验收：所有权、双向关联、拒绝非法覆盖、ClearMapTags。
 */

#include "KeyFrame.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagObservation.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

using namespace ORB_SLAM3;

namespace {

int Failures = 0;

#define EXPECT_TRUE(cond)                                                      \
    do                                                                         \
    {                                                                          \
        if(!(cond))                                                            \
        {                                                                      \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " : "       \
                      << #cond << std::endl;                                   \
            ++Failures;                                                        \
        }                                                                      \
    } while(0)

tag::TagObservation MakeObs(int tag_id, tag::CameraId cam)
{
    tag::TagObservation obs;
    obs.tag_id = tag_id;
    obs.camera_id = cam;
    return obs;
}

Map::MapTagPtr MakeTag(int id, float size = 0.16f)
{
    auto pTag = std::make_shared<tag::MapTagData>();
    pTag->SetId(id);
    pTag->SetTagSize(size);
    pTag->SetPose(Sophus::SE3f());
    pTag->SetState(tag::MapTagState::FIXED_ANCHOR);
    return pTag;
}

}  // namespace

int main()
{
    // 1) 创建 → Map::AddMapTag → KF::AddMapTag → 双向一致
    {
        Map map(0);
        KeyFrame *pKF = new KeyFrame();
        pKF->UpdateMap(&map);
        map.AddKeyFrame(pKF);
        pKF->mTagFrameData.left.push_back(MakeObs(3, tag::CameraId::LEFT_OR_MONO));
        pKF->mTagFrameData.right.push_back(MakeObs(3, tag::CameraId::RIGHT));

        auto pTag = MakeTag(3);
        EXPECT_TRUE(map.AddMapTag(pTag));
        EXPECT_TRUE(map.AddMapTag(pTag));  // 幂等
        EXPECT_TRUE(pKF->AddMapTag(pTag.get(), 0, -1));  // 仅左目
        EXPECT_TRUE(pTag->IsInKeyFrame(pKF));
        EXPECT_TRUE(pKF->GetMapTag(3) == pTag.get());
        EXPECT_TRUE(map.CheckMapTagAssociations());

        // 2) 同 MapTag 合并：补右目索引，保留左目
        EXPECT_TRUE(pKF->AddMapTag(pTag.get(), -1, 0));
        KeyFrame::MapTagAssociation assoc;
        EXPECT_TRUE(pKF->GetMapTagAssociation(3, assoc));
        EXPECT_TRUE(assoc.leftObservationIndex == 0);
        EXPECT_TRUE(assoc.rightObservationIndex == 0);
        EXPECT_TRUE(pKF->AddMapTag(pTag.get(), 0, 0));  // 完全相同：幂等
        EXPECT_TRUE(map.CheckMapTagAssociations());

        // 3) EraseMapTag：两边同时消失
        pKF->EraseMapTag(3);
        EXPECT_TRUE(pKF->GetMapTag(3) == nullptr);
        EXPECT_TRUE(!pTag->IsInKeyFrame(pKF));
        EXPECT_TRUE(map.CheckMapTagAssociations());

        // 4) 再关联后 Map::EraseMapTag 清所有 KF 引用
        EXPECT_TRUE(pKF->AddMapTag(pTag.get(), 0, 0));
        EXPECT_TRUE(map.EraseMapTag(3));
        EXPECT_TRUE(pKF->GetMapTag(3) == nullptr);
        EXPECT_TRUE(!pTag->IsInKeyFrame(pKF));
        EXPECT_TRUE(pTag->GetMap() == nullptr);
        EXPECT_TRUE(map.MapTagsInMap() == 0);

        map.EraseKeyFrame(pKF);
        delete pKF;
    }

    // 5) 同 ID 不同 MapTag 无法静默覆盖
    {
        Map map(0);
        auto a = MakeTag(1);
        auto b = MakeTag(1);
        EXPECT_TRUE(map.AddMapTag(a));
        EXPECT_TRUE(!map.AddMapTag(b));
        EXPECT_TRUE(map.GetMapTag(1).get() == a.get());
    }

    // 6) 跨 Map 关联被拒绝
    {
        Map map1(0);
        Map map2(1);
        KeyFrame *pKF = new KeyFrame();
        pKF->UpdateMap(&map2);
        map2.AddKeyFrame(pKF);
        pKF->mTagFrameData.left.push_back(MakeObs(2, tag::CameraId::LEFT_OR_MONO));

        auto pTag = MakeTag(2);
        EXPECT_TRUE(map1.AddMapTag(pTag));
        EXPECT_TRUE(!pKF->AddMapTag(pTag.get(), 0, -1));

        map2.EraseKeyFrame(pKF);
        delete pKF;
    }

    // 7) 非法索引被拒绝
    {
        Map map(0);
        KeyFrame *pKF = new KeyFrame();
        pKF->UpdateMap(&map);
        map.AddKeyFrame(pKF);
        pKF->mTagFrameData.left.push_back(MakeObs(1, tag::CameraId::LEFT_OR_MONO));
        auto pTag = MakeTag(1);
        EXPECT_TRUE(map.AddMapTag(pTag));
        EXPECT_TRUE(!pKF->AddMapTag(pTag.get(), 5, -1));
        EXPECT_TRUE(!pKF->AddMapTag(pTag.get(), -1, -1));
        EXPECT_TRUE(map.CheckMapTagAssociations());
        map.EraseKeyFrame(pKF);
        delete pKF;
    }

    // 8) ClearMapTags：无悬空裸指针
    {
        Map map(0);
        KeyFrame *pKF = new KeyFrame();
        pKF->UpdateMap(&map);
        map.AddKeyFrame(pKF);
        pKF->mTagFrameData.left.push_back(MakeObs(9, tag::CameraId::LEFT_OR_MONO));
        auto pTag = MakeTag(9);
        EXPECT_TRUE(map.AddMapTag(pTag));
        EXPECT_TRUE(pKF->AddMapTag(pTag.get(), 0, -1));

        map.ClearMapTags();
        EXPECT_TRUE(map.MapTagsInMap() == 0);
        EXPECT_TRUE(pKF->GetMapTag(9) == nullptr);
        EXPECT_TRUE(!pTag->IsInKeyFrame(pKF));
        EXPECT_TRUE(pTag->GetMap() == nullptr);
        EXPECT_TRUE(map.CheckMapTagAssociations());

        map.EraseKeyFrame(pKF);
        delete pKF;
    }

    // 9) 世界角点按需计算
    {
        auto pTag = MakeTag(0, 0.2f);
        pTag->SetPose(Sophus::SE3f());
        EXPECT_TRUE(pTag->HasWorldCorners());
        const auto c = pTag->GetWorldCorners();
        EXPECT_TRUE(std::abs(c[0].x() + 0.1f) < 1e-5f);
        EXPECT_TRUE(std::abs(c[0].y() - 0.1f) < 1e-5f);
    }

    // 10) Check 能报告失败原因
    {
        Map map(0);
        KeyFrame *pKF = new KeyFrame();
        pKF->UpdateMap(&map);
        map.AddKeyFrame(pKF);
        pKF->mTagFrameData.left.push_back(MakeObs(1, tag::CameraId::LEFT_OR_MONO));
        auto pTag = MakeTag(1);
        map.AddMapTag(pTag);
        pKF->AddMapTag(pTag.get(), 0, -1);
        // 人为破坏：清空测量使索引失效
        pKF->mTagFrameData.left.clear();
        std::string err;
        EXPECT_TRUE(!map.CheckMapTagAssociations(&err));
        EXPECT_TRUE(!err.empty());

        pKF->EraseAllMapTags();
        map.EraseMapTag(1);
        map.EraseKeyFrame(pKF);
        delete pKF;
    }

    if(Failures != 0)
    {
        std::cerr << Failures << " expectation(s) failed" << std::endl;
        return 1;
    }
    std::cout << "test_map_tag_associations: OK" << std::endl;
    return 0;
}
