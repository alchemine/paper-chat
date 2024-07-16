"""Configuration files"""

from os.path import join, dirname, abspath

from paper_chat.core.utils import load_yaml


############################################################
# Paths
############################################################
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
CONFIGS_PATH = join(ROOT_PATH, "configs")


############################################################
# Load configurations
############################################################
_CONFIGS_LLM_PATH = join(CONFIGS_PATH, "llm.yml")
CONFIGS_LLM = load_yaml(_CONFIGS_LLM_PATH)
