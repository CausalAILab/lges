#pragma once

#include <Eigen/Dense>
#include <string>

/** Load a 2-D float64 .npy matrix with at least two rows and one column.
 * Supports C/Fortran order, either byte order, and NPY versions 1.0–3.0.
 * Rejects unsupported dtypes, malformed shapes/headers and truncated payloads.
 * The result owns its memory; BICScorer separately validates numerical inputs.
 */
Eigen::MatrixXd load_npy_data(const std::string &filename);
