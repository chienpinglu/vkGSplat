# vkSplat unit tests.
#
# We avoid pulling in a heavyweight test framework in v0 — each test is
# a tiny standalone executable that returns nonzero on failure and is
# registered with CTest. Once the surface area justifies it we will
# switch to GoogleTest or doctest.

function(vksplat_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vksplat)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

vksplat_add_test(test_camera)
vksplat_add_test(test_scene)
vksplat_add_test(test_types)
