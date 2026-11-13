# BridgeAltTests.cmake
# ---------------------------------------------------------------------------
# Build the standalone test_alt_* programs (flat *.cpp-in-root layout, or the
# src/ + src/Tests/ layout) and link them against the bridge library.  Included
# from the root CMakeLists.txt AFTER add_subdirectory(src)/add_subdirectory(tests)
# so `bridge` already exists.

file(READ "${CMAKE_SOURCE_DIR}/VERSION" _BRIDGE_VERSION)
string(STRIP "${_BRIDGE_VERSION}" _BRIDGE_VERSION)

foreach(_dir IN ITEMS
        test_alt_spectrum
        test_alt_smear
        test_alt_hmc)
    set(_root_dir "${CMAKE_SOURCE_DIR}/${_dir}")
    if(NOT IS_DIRECTORY "${_root_dir}")
        continue()
    endif()

    # Flat layout: *.cpp directly in the test directory
    file(GLOB _srcs CONFIGURE_DEPENDS "${_root_dir}/*.cpp")
    if(_srcs)
        set(_inc_dir "${_root_dir}")
    else()
        # src/ layout: test_alt_spectrum, test_alt_smear, test_alt_hmc
        file(GLOB _srcs CONFIGURE_DEPENDS
            "${_root_dir}/src/*.cpp"
            "${_root_dir}/src/Tests/*.cpp")
        set(_inc_dir "${_root_dir}/src")
    endif()

    if(NOT _srcs)
        continue()
    endif()

    string(REPLACE "test_alt_" "" _short "${_dir}")
    add_executable(bridge_test_alt_${_short} ${_srcs})
    target_include_directories(bridge_test_alt_${_short} PRIVATE "${_inc_dir}")
    target_compile_definitions(bridge_test_alt_${_short} PRIVATE
        "_BRIDGE_VERSION=\"${_BRIDGE_VERSION}\"")
    target_link_libraries(bridge_test_alt_${_short} PRIVATE bridge)
endforeach()

