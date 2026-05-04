# vksplat_viewer — debug viewer for the paused 3DGS path.
#
# Headless by default; pass --present to open a swapchain (requires
# the host system to have a windowing surface, which the SDG path does
# not need).

if(VKSPLAT_ENABLE_3DGS)
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
else()
    message(STATUS "vksplat_viewer disabled (VKSPLAT_ENABLE_3DGS=OFF)")
endif()
