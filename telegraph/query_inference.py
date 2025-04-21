import bittensor as bt
from typing import Any, Optional, Dict

async def query_subnet_model(
    wallet: "bt.wallet",
    axon_info: "bt.axon_info",
    inference_code: str,
    data: Any,
    timeout: float = 12.0
) -> Dict[str, Any]:
    """
    Query a model on another subnet through the Telegraph protocol
    """
    try:
        dendrite = bt.dendrite(wallet=wallet)
        synapse = bt.Synapse()
        synapse.data = {
            "inference_code": inference_code,
            "payload": data
        }
        response = await dendrite(
            axons=[axon_info],
            synapse=synapse,
            deserialize=True,
            timeout=timeout
        )
        if hasattr(response, "error") and response.error:
            return {"error": response.error}
        return {"response": getattr(response, "response", None)}
    except Exception as e:
        bt.logging.error(f"Failed to query model: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}