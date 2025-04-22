import asyncio
from neurons.validator.utils.performance_calculator import PerformanceCalculator
from base.types import ChainType, TokenPrediction
from datetime import datetime

async def test_performance_calculator():
    calculator = PerformanceCalculator()
    
    # Create some mock predictions
    predictions = [
        TokenPrediction(
            chain=ChainType.BASE,
            addresses=["0x123", "0x456"],
            pairAddresses=["0xpair1", "0xpair2"],
            timestamp=datetime.now(),
            confidence_scores={"0x123": 0.8, "0x456": 0.7}
        )
    ]
    
    miner_uids = [2]  # Your miner UID
    
    # Test performance calculation
    performances, uids = await calculator.calculate_performance(predictions, miner_uids)
    
    print(f"Performances: {performances}")
    print(f"UIDs: {uids}")

if __name__ == "__main__":
    asyncio.run(test_performance_calculator())
