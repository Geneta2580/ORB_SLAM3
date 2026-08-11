#include "OptimizableTypes.h"

#include <iostream>

int main()
{
    double max_err = 0.0;
    const bool ok = ORB_SLAM3::TestTagCornerEdgeJacobians(&max_err);
    if(!ok)
    {
        std::cerr << "Tag corner edge Jacobian check FAILED, max_abs_err="
                  << max_err << std::endl;
        return 1;
    }
    std::cout << "Tag corner edge Jacobian check OK, max_abs_err=" << max_err
              << std::endl;
    return 0;
}
