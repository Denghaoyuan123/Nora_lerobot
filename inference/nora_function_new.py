#### read the joints & get front images ##########################################
/home/luka/Nora_lerobot/lerobot/src/lerobot/record.py
line:235 observation = robot.get_observation()

-->/home/luka/Nora_lerobot/lerobot/src/lerobot/robots/robot.py
line:156 def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state. Its structure
                should match :pymeth:`observation_features`.
        """

        pass
    
#### send the joints #############################################################
/home/luka/Nora_lerobot/lerobot/src/lerobot/record.py
line:271 sent_action = robot.send_action(action)

-->/home/luka/Nora_lerobot/lerobot/src/lerobot/robots/robot.py
line:168 def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass


#### about how to get ####################################
        observation = robot.get_observation()

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and isinstance(teleop, Teleoperator):
            action = teleop.get_action()
        elif policy is None and isinstance(teleop, list):
            # TODO(pepijn, steven): clean the record loop for use of multiple robots (possibly with pipeline)
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)
####about how to get



##############################   D E M O ######################################
from lerobot.robots import make_robot_from_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

# 1) 构造 so101_follower + front 相机
robot_cfg = SO101FollowerConfig(
    type="so101_follower",
    id="my_follower",
    port="/dev/ttyACM5",
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=0, width=640, height=480, fps=15,
            color_mode="rgb"  # dataclass 要求；无旋转就不填或按枚举
        )
    },
)
robot = make_robot_from_config(robot_cfg)   # 工厂按 type 选择具体类（so101_follower）:contentReference[oaicite:9]{index=9}
robot.connect()                              # 会连相机、做必要校准、配置等:contentReference[oaicite:10]{index=10}

# 2) —— 读取 “关节 + front 图像” ——（全在一个字典）
obs = robot.get_observation()                # 关节: "<motor>.pos"；相机: "front" 键:contentReference[oaicite:11]{index=11}
qpos = [obs[k] for k in robot.action_features]  # 关节顺序 = robot.action_features
img  = obs["front"]                            # front 相机帧

# 3) —— 下发动作 ——（dict 形式，键名沿用 "<motor>.pos"）
target = {k: 0.0 for k in robot.action_features}    # 举例：全 0
sent   = robot.send_action(target)                  # 返回“实际发送的动作”:contentReference[oaicite:12]{index=12}

robot.disconnect()
##############################   D E M O ######################################