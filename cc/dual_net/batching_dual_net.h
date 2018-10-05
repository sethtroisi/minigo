#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <memory>

#include "cc/dual_net/factory.h"

namespace minigo {

std::unique_ptr<DualNetFactory> NewBatchingFactory(
    std::unique_ptr<DualNet> dual_net);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_
