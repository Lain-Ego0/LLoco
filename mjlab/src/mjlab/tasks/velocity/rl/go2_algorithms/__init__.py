"""Go2-specific learning components used by the migrated custom tasks.

The modules in this package deliberately do not depend on the legacy Isaac Gym
``rsl_rl`` fork.  They expose small PyTorch building blocks that can be used by
the current mjlab runner and by headless smoke tests.
"""

from .algorithms import (
  AmpAlgorithm,
  AmpCtsPPO,
  AmpDreamWaQPPO,
  AmpPPO,
  AmpTeacherStudentPPO,
  CtsAlgorithm,
  CtsPPO,
  DreamWaQAlgorithm,
  DreamWaQPPO,
  Go2AuxiliaryPPO,
  TeacherStudentAlgorithm,
  TeacherStudentPPO,
)
from .deployment import (
  Go2DeploymentAdapter,
  Go2HistoryBuffer,
  Go2OnnxPolicy,
  Go2PolicyInputSpec,
  Go2RecurrentDeploymentAdapter,
  Go2RecurrentOnnxPolicy,
)
from .models import (
  AmpDiscriminator,
  CtsActorCritic,
  CtsActorModel,
  CtsCriticModel,
  CtsStudentActorModel,
  DreamWaQActorCritic,
  DreamWaQActorModel,
  StudentActorModel,
  TeacherActorModel,
  TeacherStudentActorCritic,
)
from .motion import (
  GO2_JOINT_NAMES,
  Go2MotionLoader,
  Go2MotionTrajectory,
  joint_permutation,
)
from .runner import Go2CustomRunner
from .storage import (
  Go2AmpReplayBuffer,
  Go2RolloutStorage,
  Go2RunningNormalizer,
  Go2Transition,
)

__all__ = [
  "AmpDiscriminator",
  "AmpAlgorithm",
  "AmpCtsPPO",
  "AmpDreamWaQPPO",
  "AmpPPO",
  "AmpTeacherStudentPPO",
  "CtsActorCritic",
  "CtsActorModel",
  "CtsCriticModel",
  "CtsStudentActorModel",
  "CtsPPO",
  "CtsAlgorithm",
  "DreamWaQActorCritic",
  "DreamWaQActorModel",
  "DreamWaQAlgorithm",
  "DreamWaQPPO",
  "Go2AuxiliaryPPO",
  "Go2RolloutStorage",
  "Go2RunningNormalizer",
  "Go2AmpReplayBuffer",
  "Go2MotionLoader",
  "Go2MotionTrajectory",
  "GO2_JOINT_NAMES",
  "joint_permutation",
  "Go2DeploymentAdapter",
  "Go2HistoryBuffer",
  "Go2OnnxPolicy",
  "Go2PolicyInputSpec",
  "Go2RecurrentDeploymentAdapter",
  "Go2RecurrentOnnxPolicy",
  "Go2Transition",
  "Go2CustomRunner",
  "TeacherStudentActorCritic",
  "TeacherActorModel",
  "StudentActorModel",
  "TeacherStudentAlgorithm",
  "TeacherStudentPPO",
]
