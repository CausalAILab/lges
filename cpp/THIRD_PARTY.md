# Source and dependency attribution

The C++ graph/search implementation derives from [ANazaret/XGES](https://github.com/ANazaret/XGES)
by Achille Nazaret, associated with Achille Nazaret and David Blei's XGES work.
Original author notices remain in the source. The upstream repository does not
currently provide a repository-level license file; this project does not invent
or replace an upstream license grant.

Safe/Conservative insertion and the LGES-family schedules are inspired by
[CausalAILab/lges](https://github.com/CausalAILab/lges), by Adiba Ejaz and Elias
Bareinboim. Test/experiment utilities can import that checkout explicitly.

- `lib/cxxopts.hpp`: Jarryd Beck and contributors; the MIT notice is preserved in the header.
- spdlog: Gabi Melman and contributors; fetched at commit
  `6fa36017cfd5731d617e1a934f0e5ea9c4445b13`, retaining its MIT license and bundled
  fmt notices in the downloaded source. The archive checksum is pinned in `lib/CMakeLists.txt`.
- Eigen, Boost and BLAS are build dependencies, not vendored source in this project.

Copying this project under `cpp/` imports files, not the original Git history.
Generated build trees, runtime environments and experiment outputs are excluded.
