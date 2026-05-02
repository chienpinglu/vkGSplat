# vksplat_viewer — debug viewer for the v1 3DGS path.
#
# Headless by default; pass --present to open a swapchain (requires
# the host system to have a windowing surface, which the SDG path does
# not need).

add_executable(vksplat_viewer
    main.cpp
)

target_link_libraries(vksplat_viewer
    PRIVATE
        vksplat
)

set_target_properties(vksplat_viewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/viewer"
)
