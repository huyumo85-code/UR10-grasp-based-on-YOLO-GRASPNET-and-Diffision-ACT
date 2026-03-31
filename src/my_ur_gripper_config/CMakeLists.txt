cmake_minimum_required(VERSION 3.8)
project(my_ur_gripper_config)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 声明依赖
find_package(ament_cmake REQUIRED)

# 2. 【核心修复】使用更明确的路径安装方式
# 确保包含 urdf, meshes, config 等所有资源文件夹（如果你有的话）
install(
  DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# 3. 必须在 ament_package() 之前导出路径索引
# 这能确保 ros2 launch 能通过包名找到文件
ament_export_dependencies(ament_cmake)

ament_package()
