from enum import Enum


class AgentType(str, Enum):
    qa = "qa"
    instruction_following = "instruction_following"
    domain_specialist = "domain_specialist"
    style = "style"
    adversarial = "adversarial"
    conversation = "conversation"

