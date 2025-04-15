import bittensor as bt
from typing import Any, Optional, Dict
from telegraph.protocol import InferenceRequestSynapse

async def query_subnet_model(
    wallet: "bt.wallet",
    axon_info: "bt.axon_info",
    inference_code: str,
    data: Any,
    timeout: float = 12.0
) -> Dict[str, Any]:
    """
    Query a model on another subnet through the Telegraph protocol
    
    Args:
        wallet: Bittensor wallet for signing requests
        axon_info: Validator axon info to query
        inference_code: The code identifying which model to use
        data: The data to process with the model
        timeout: Maximum time to wait for response in seconds
        
    Returns:
        Dict containing either "response" with model output or "error" with error message
    """
    try:
        # Create dendrite client
        dendrite = bt.dendrite(wallet=wallet)
        
        # Create synapse with request data
        synapse = InferenceRequestSynapse(
            inference_code=inference_code,
            data=data
        )
        
        # Send request to validator
        response = await dendrite(
            axons=[axon_info],
            synapse=synapse,
            deserialize=True,
            timeout=timeout
        )
        
        # Check for errors
        if hasattr(response, "error") and response.error:
            return {"error": response.error}
        
        # Return successful response
        return {"response": response.response}
        
    except Exception as e:
        bt.logging.error(f"Failed to query model: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}