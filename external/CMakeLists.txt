add_library(eigen3 INTERFACE IMPORTED GLOBAL)
target_include_directories(eigen3 INTERFACE eigen)
add_subdirectory(tinyply)

add_subdirectory(args)
add_library(glm INTERFACE IMPORTED GLOBAL)
target_include_directories(glm INTERFACE glm)
