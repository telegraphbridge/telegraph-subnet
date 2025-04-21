# add_telegraph_subnet.py
import sys
import os
from telegraph.registry import InferenceRegistry
import json

# Create a custom registry file
telegraph_registry = {
    "telegraph": {
        "netuid": 349, 
        "description": "Telegraph cross-chain protocol"
    }
}

# Save to file
with open("telegraph_registry.json", "w") as f:
    json.dump(telegraph_registry, f, indent=2)
print(f"Created custom registry file: telegraph_registry.json")

# Load the registry with custom file
registry = InferenceRegistry(config_path="telegraph_registry.json")

# Verify telegraph is in registry
print("\nVerifying Telegraph subnet in registry:")
all_codes = registry.get_all_codes()
print(f"Registry contains {len(all_codes)} codes")
print(f"Is 'telegraph' in registry? {'telegraph' in all_codes}")

# Get codes for your subnet
telegraph_codes = registry.get_codes_for_subnet(349)
print(f"Codes for subnet 349: {telegraph_codes}")