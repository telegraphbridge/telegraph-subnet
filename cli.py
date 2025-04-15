#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import argparse
import bittensor as bt

# Add project root to path to allow importing telegraph module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telegraph.client import TelegraphClient

async def main():
    parser = argparse.ArgumentParser(description="Telegraph Cross-Subnet Inference Client")
    parser.add_argument("--wallet", type=str, required=True, help="Wallet name")
    parser.add_argument("--hotkey", type=str, required=True, help="Hotkey name")
    parser.add_argument("--netuid", type=int, default=1, help="Telegraph subnet netuid")
    parser.add_argument("--list-validators", action="store_true", help="List available validators")
    parser.add_argument("--list-codes", action="store_true", help="List available inference codes")
    parser.add_argument("--code", type=str, help="Inference code (e.g., gpt3)")
    parser.add_argument("--validator", type=int, help="Validator UID to query")
    parser.add_argument("--data", type=str, help="JSON data string or @file.json")
    args = parser.parse_args()
    
    # Load wallet
    wallet = bt.wallet(name=args.wallet, hotkey=args.hotkey)
    
    # Create client
    client = TelegraphClient(wallet=wallet, telegraph_netuid=args.netuid)
    
    # List validators
    if args.list_validators:
        validators = client.list_validators()
        print("Available Telegraph validators:")
        for v in validators:
            print(f"UID: {v['uid']}, Hotkey: {v['hotkey']}, Stake: {v['stake']}")
        return
        
    # List inference codes
    if args.list_codes:
        codes = client.get_available_codes()
        print("Available inference codes:")
        for code in codes:
            print(f"- {code}")
        return
    
    # Check required arguments for query
    if not args.code:
        print("Error: --code argument required for inference requests")
        return
        
    if not args.data:
        print("Error: --data argument required for inference requests")
        return
    
    # Parse data
    if args.data.startswith('@'):
        # Load from file
        with open(args.data[1:], 'r') as f:
            data = json.load(f)
    else:
        # Parse as JSON string
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            print("Error: --data must be valid JSON or path to JSON file (prefix with @)")
            return
    
    # Run inference query
    print(f"Querying model with code '{args.code}'...")
    result = await client.query_model(
        inference_code=args.code,
        data=data,
        validator_uid=args.validator
    )
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())