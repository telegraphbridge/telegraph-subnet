# test_registry.py
import sys
import os
from telegraph.registry import InferenceRegistry

# Create instance of registry
registry = InferenceRegistry()

# Print available inference codes
print(f"Registry contains {len(registry.get_all_codes())} inference codes")
print("\nSample codes:")
for code in list(registry.get_all_codes())[:5]:
    netuid = registry.get_netuid(code)
    print(f"  Code: {code} -> Subnet: {netuid}")

# Test code validation
test_codes = ["root", "apex", "invalid_code", "cortex"]
print("\nTesting code validation:")
for code in test_codes:
    is_valid = registry.is_valid_code(code)
    print(f"  Is '{code}' valid? {is_valid}")

# Find codes for specific subnets
test_netuids = [0, 1, 18, 349]
print("\nFinding codes for specific subnets:")
for netuid in test_netuids:
    codes = registry.get_codes_for_subnet(netuid)
    print(f"  Subnet {netuid} has {len(codes)} codes: {codes}")