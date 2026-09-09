from pathlib import Path
ROOT = Path(__file__).parent
ROBOT_XML_DICT = {"unitree_g1": ROOT / "assets/g1_mocap_29dof.xml"}
IK_CONFIG_DICT = {"bvh": {"unitree_g1": ROOT / "bvh_to_g1.json"}}
