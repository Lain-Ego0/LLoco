"""Lightweight catalog; importing the workbench never imports torch."""

TASKS = {
  "velocity": [
    "Unitree-Go2-Flat",
    "Unitree-Go2-Rough",
    "Unitree-A2-Flat",
    "Unitree-As2-Flat",
    "Unitree-G1-Flat",
    "Unitree-G1-23Dof-Flat",
    "Unitree-H1_2-Flat",
    "Unitree-H2-Flat",
    "Unitree-R1-Flat",
  ],
  "tracking": ["Unitree-G1-Tracking", "Unitree-G1-23Dof-Tracking"],
}
TASK_ASSETS = {
  "Unitree-Go2-Flat": "src/lloco/assets/robots/unitree_go2/xmls/scene_go2.xml",
  "Unitree-Go2-Rough": "src/lloco/assets/robots/unitree_go2/xmls/scene_go2.xml",
  "Unitree-A2-Flat": "src/lloco/assets/robots/unitree_a2/xmls/scene_a2.xml",
  "Unitree-As2-Flat": "src/lloco/assets/robots/unitree_as2/xmls/as2.xml",
  "Unitree-G1-Flat": "src/lloco/assets/robots/unitree_g1/xmls/scene_g1.xml",
  "Unitree-G1-23Dof-Flat": "src/lloco/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml",
  "Unitree-H1_2-Flat": "src/lloco/assets/robots/unitree_h1_2/xmls/h1_2.xml",
  "Unitree-H2-Flat": "src/lloco/assets/robots/unitree_h2/xmls/h2.xml",
  "Unitree-R1-Flat": "src/lloco/assets/robots/unitree_r1/xmls/r1.xml",
  "Unitree-G1-Tracking": "src/lloco/assets/robots/unitree_g1/xmls/scene_g1.xml",
  "Unitree-G1-23Dof-Tracking": "src/lloco/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml",
}

SKILLS = {
  "PPO · Trot": "Unitree-Go2-Trot-Flat",
  "PPO · Jump": "Unitree-Go2-Jump-Flat",
  "PPO · Rear Stand": "Unitree-Go2-Rear-Stand-Flat",
  "PPO · Hand Stand": "Unitree-Go2-Handstand-Flat",
  "PPO · Spring Jump": "Unitree-Go2-Spring-Jump-Flat",
  "DreamWaQ": "Unitree-Go2-DreamWaQ-Rough",
  "AMP-DreamWaQ": "Unitree-Go2-AMP-DreamWaQ-Rough",
  "CTS": "Unitree-Go2-CTS-Rough",
  "TS": "Unitree-Go2-TS-Teacher-Rough",
}
ALGORITHMS = {
  "velocity": {"PPO": TASKS["velocity"]},
  "tracking": {"PPO · Motion Tracking": TASKS["tracking"]},
  "go2_skill": {name: [task] for name, task in SKILLS.items()},
}
for task in SKILLS.values():
  TASK_ASSETS[task] = TASK_ASSETS["Unitree-Go2-Flat"]
