if(SOPHUS_INCLUDE_DIR)
  set(SOPHUS_FOUND TRUE)
else()
  find_path(
    SOPHUS_INCLUDE_DIR
    NAMES sophus/se3.hpp
    PATHS ${CMAKE_INSTALL_PREFIX}/include ${CMAKE_CURRENT_SOURCE_DIR}/../sophus
    PATH_SUFFIXES sophus)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Sophus DEFAULT_MSG SOPHUS_INCLUDE_DIR)
  mark_as_advanced(SOPHUS_INCLUDE_DIR)
endif()
