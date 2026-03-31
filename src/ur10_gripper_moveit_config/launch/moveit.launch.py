import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():
    
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation clock if true"
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # 1. 构建 MoveIt 配置
    moveit_config = (
        MoveItConfigsBuilder("ur10_with_robotiq", package_name="ur10_gripper_moveit_config")
        .robot_description(file_path="config/ur10_with_robotiq.urdf.xacro")
        .robot_description_semantic(file_path="config/ur10_with_robotiq.srdf")
        
        
        # 因为我们在下面手动注入
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # 2. 准备 MoveGroup 参数
    move_group_params = moveit_config.to_dict()
    # ====== ✨ 手动注入传感器配置，绕过 Builder 的 Bug ======
    import yaml
    sensors_path = os.path.join(
        get_package_share_directory("ur10_gripper_moveit_config"),
        "config",
        "sensors_3d.yaml",
    )
    with open(sensors_path, "r") as f:
        sensors_config = yaml.safe_load(f)
    
    move_group_params.update(sensors_config)
    # ========================================================

    moveit_controllers = {
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller", "robotiq_gripper_controller"],
            "joint_trajectory_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
                ]
            },
            "robotiq_gripper_controller": {
                "type": "GripperCommand",
                "action_ns": "gripper_cmd",
                "default": True,
                "joints": ["robotiq_85_left_knuckle_joint"]
            }
        },
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    
    # 合并控制器配置
    move_group_params.update(moveit_controllers)
    
    # 注入性能参数
    move_group_params.update({
        "use_sim_time": use_sim_time,
        "publish_robot_description_semantic": True,
        "max_velocity_scaling_factor": 1.0,
        "max_acceleration_scaling_factor": 1.0,
    })

    # 3. MoveGroup 节点
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            move_group_params,
            {"sensors": ["PointCloudOctomapUpdater"]}, 
            {
                "PointCloudOctomapUpdater": {
                    "sensor_plugin": "occupancy_map_monitor/PointCloudOctomapUpdater",
                    "point_cloud_topic": "/d435/points",  # 👈 改成你实际的话题
                    "max_range": 3.0,                     # 适当加大范围
                    "best_effort": True,
                    "fixed_frame": "world",               # 👈 显式指定参考坐标系
                    "filtered_cloud_topic": "filtered_cloud",
                }
            },
        ],
    )

    # 4. RViz 节点
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur10_gripper_moveit_config"), "config", "official_style.rviz"]
    )

    run_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        run_move_group_node,
        run_rviz_node,
    ])
