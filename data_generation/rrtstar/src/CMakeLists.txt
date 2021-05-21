SET(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig:/opt/local/lib/pkgconfig:/usr/local/share/pkgconfig")

find_package(nlohmann_json 3.2.0 REQUIRED)

add_compile_options(-std=c++17)
pods_install_pkg_config_file(rrtstar
    CFLAGS
    LIBS  -lrrtstar
    REQUIRES lcmtypes
    VERSION 0.0.1)

include_directories(
    ${LCM_INCLUDE_DIRS})

find_package(google_cloud_cpp_storage REQUIRED)
add_executable(rrtstar rrts_main.cpp system_single_integrator.cpp kdtree.c )
add_executable(timing timing.cpp system_single_integrator.cpp kdtree.c)

pods_use_pkg_config_packages(rrtstar
    bot2-core
    lcmtypes)

pods_install_executables(rrtstar timing)

target_link_libraries(rrtstar -llcm google-cloud-cpp::storage -lstdc++fs)
target_link_libraries(timing -llcm nlohmann_json::nlohmann_json -lstdc++fs)
