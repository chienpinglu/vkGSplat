# vkgsplat_viewer — debug viewer for the paused 3DGS path.
#
# Headless by default; pass --present to open a swapchain (requires
# the host system to have a windowing surface, which the SDG path does
# not need).

if(VKGSPLAT_ENABLE_3DGS)
    add_executable(vkgsplat_viewer
        main.cpp
    )

    target_link_libraries(vkgsplat_viewer
        PRIVATE
            vkgsplat
    )

    set_target_properties(vkgsplat_viewer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/viewer"
    )
else()
    message(STATUS "vkgsplat_viewer disabled (VKGSPLAT_ENABLE_3DGS=OFF)")
endif()
