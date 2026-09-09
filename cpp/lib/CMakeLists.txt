# Keep the logging and bundled formatting implementation fixed across builds.
if(NOT TARGET spdlog::spdlog)
    include(FetchContent)
    FetchContent_Declare(xges_spdlog
        URL https://codeload.github.com/gabime/spdlog/tar.gz/6fa36017cfd5731d617e1a934f0e5ea9c4445b13
        URL_HASH SHA256=5097fb362e79a2bd7247beaf1f8377ed60e274fbe83a4b33e7b73383f0279022
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(xges_spdlog)
endif()
