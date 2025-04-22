import sys
import os
import bittensor as bt
import json
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse

# Test PredictionSynapse serialization
prediction = PredictionSynapse(
    chain_name="BASE",
    addresses=["0x123", "0x456"],
    pairAddresses=["0xPair1", "0xPair2"]
)
print("\nPrediction Synapse:")
print(f"  Chain: {prediction.chain_name}")
print(f"  Addresses: {prediction.addresses}")
print(f"  Pairs: {prediction.pairAddresses}")

# Use our custom to_json method as in the original
serialized = prediction.to_json()
print(f"\nSerialized: {serialized[:100]}...")  # Show beginning of serialized data

# Deserialize back
deserialized = PredictionSynapse().from_json(serialized)
print(f"\nDeserialized correctly? {deserialized.chain_name == prediction.chain_name and deserialized.addresses == prediction.addresses}")

# Test InferenceRequestSynapse
inference = InferenceRequestSynapse(
    inference_code="cortex",
    data={"prompt": "Hello world"}
)
print("\nInference Request Synapse:")
print(f"  Code: {inference.inference_code}")
print(f"  Data: {inference.data}")

# Serialize inference request
inference_serialized = inference.to_json()
print(f"\nSerialized inference: {inference_serialized[:100]}...")

# Test serializing with binary format for network transmission
binary_serialized = prediction.serialize()
print(f"\nBinary serialized length: {len(binary_serialized)} bytes")

# Deserialize from binary
binary_deserialized = PredictionSynapse().deserialize(binary_serialized)
print(f"\nBinary deserialization successful: {binary_deserialized.chain_name == prediction.chain_name}")