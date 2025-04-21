from typing import Dict
import os
from dotenv import load_dotenv
from base.types import ChainType, LiquidityMetrics
from web3 import Web3

class LiquidityChecker:
    def __init__(self):
        self.web3_connections: Dict[ChainType, Web3] = {}
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize blockchain connections"""
        load_dotenv()
        base_rpc_url = os.getenv('BASE_RPC_URL')
        if not base_rpc_url:
            raise ValueError("BASE_RPC_URL not found in environment variables")
        
        self.web3_connections[ChainType.BASE] = Web3(Web3.HTTPProvider(base_rpc_url))

    async def check_token_liquidity(
        self, 
        chain: ChainType, 
        token_address: str,
        token_pair: str
    ) -> LiquidityMetrics:
        """Check current liquidity metrics for a token"""
        if chain != ChainType.BASE:
            raise ValueError("Only Base L2 chain is currently supported")
        
        web3 = self.web3_connections[chain]
        
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        # Get token contract
        token_contract = web3.eth.contract(
            address=Web3.to_checksum_address(token_pair),
            abi=erc20_abi
        )
        
        # Get liquidity amount
        liquidity = token_contract.functions.balanceOf(
            Web3.to_checksum_address(token_address)
        ).call()
        
        # Convert from wei to ether for readability
        liquidity_in_ether = Web3.from_wei(liquidity, 'ether')
        
        return LiquidityMetrics(current_liquidity=float(liquidity_in_ether)) 