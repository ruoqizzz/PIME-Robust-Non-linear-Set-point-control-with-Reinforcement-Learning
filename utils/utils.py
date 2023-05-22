from elegantrl.agent import AgentSAC, AgentTD3, AgentPPO
from elegantrl.agent_residual import AgentResidualIntegratorModularPPO, AgentResidualPPO


MODELS = {
	"td3": AgentTD3,
    "ppo": AgentPPO,
	"sac": AgentSAC,
	"residualintegratormodularppo": AgentResidualIntegratorModularPPO,
	"residualppo": AgentResidualPPO,
}

IF_ONPOLICY = {
	"td3": False,
    "ppo": True,
	"sac": False,
	"residualintegratormodularppo": True,
	"residualppo": True
}
