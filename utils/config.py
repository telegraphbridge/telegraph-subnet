import argparse
import bittensor as bt

def add_args(parser: argparse.ArgumentParser):
    """Add Telegraph subnet specific arguments to parser"""
    # Add custom Telegraph args
    return parser

def check_config(config: bt.config):
    """Check and validate Telegraph subnet config"""
    # Validation logic
    return config

def config():
    """Get default Telegraph subnet config"""
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser = add_args(parser)
    return bt.config(parser)
