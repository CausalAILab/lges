//
// Created by Achille Nazaret on 11/20/23.
//

#include "set_ops.h"

void union_with_single_element(const FlatSet &a, int b, FlatSet &out) {
    out = a;
    out.insert(b);
}
