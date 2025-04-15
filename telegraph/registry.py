import bittensor as bt
from typing import Dict, Optional, List, Set

class InferenceRegistry:
    """Registry mapping inference codes to subnet information"""
    
    def __init__(self, config_path: str = None):
        """Initialize registry with default inference codes
        
        Args:
            config_path: Optional path to config file with registered codes
        """
        # Default registry: "code": {"netuid": subnet_id, "description": description}
        self.registry = {
            "root": {"netuid": 0, "description": "τ root"},
            
            # Other subnets in order of netuid
            "apex": {"netuid": 1, "description": "α apex"},
            "templar": {"netuid": 3, "description": "γ templar"},
            "targon": {"netuid": 4, "description": "δ targon"},
            "pretrain": {"netuid": 9, "description": "ι pretrain"},
            "oracle": {"netuid": 28, "description": "ד oracle"},
            "dataverse": {"netuid": 13, "description": "ν dataverse"},
            "palaidn": {"netuid": 14, "description": "ξ palaidn"},
            "deval": {"netuid": 15, "description": "ο deval"},
            "cortex": {"netuid": 18, "description": "σ cortex"},
            "any-any": {"netuid": 21, "description": "φ any-any"},
            "protein": {"netuid": 25, "description": "א protein"},
            "audio": {"netuid": 50, "description": "ש audio"},
            "bani": {"netuid": 98, "description": "ბ bani"},
            "gani": {"netuid": 99, "description": "გ gani"},
            "alfa": {"netuid": 117, "description": "Ⲁ alfa"},
            "alfas": {"netuid": 118, "description": "ⲁ alfas"},
            "vida": {"netuid": 119, "description": "Ⲃ vida"},
            "glagolitic_uku": {"netuid": 155, "description": "Ⱅ glagolitic_uku"},
            "glagolitic_ja": {"netuid": 166, "description": "Ⱐ glagolitic_ja"},
            "thai_kho_rakhang": {"netuid": 171, "description": "ฅ thai_kho_rakhang"},
            "hangul_nieun": {"netuid": 212, "description": "ᄁ hangul_nieun"},
            "hangul_yeo": {"netuid": 231, "description": "ᅧ hangul_yeo"},
            "hangul_o": {"netuid": 233, "description": "ᅩ hangul_o"},
            "hangul_wa": {"netuid": 234, "description": "ᅪ hangul_wa"},
            "hangul_wae": {"netuid": 235, "description": "ᅫ hangul_wae"},
            "hangul_eo": {"netuid": 229, "description": "ᅥ hangul_eo"},
            "hangul_u": {"netuid": 238, "description": "ᅮ hangul_u"},
            "hangul_weo": {"netuid": 239, "description": "ᅯ hangul_weo"},
            "hangul_we": {"netuid": 240, "description": "ᅰ hangul_we"},
            "hangul_wi": {"netuid": 241, "description": "ᅱ hangul_wi"},
            "hangul_yu": {"netuid": 242, "description": "ᅲ hangul_yu"},
            "hangul_eu": {"netuid": 243, "description": "ᅳ hangul_eu"},
            "hangul_i": {"netuid": 245, "description": "ᅵ hangul_i"},
            "ethiopic_glottal_a": {"netuid": 246, "description": "አ ethiopic_glottal_a"},
            "synth": {"netuid": 247, "description": "ኡ Synth"},
            "ethiopic_glottal_i": {"netuid": 248, "description": "ኢ ethiopic_glottal_i"},
            "ethiopic_glottal_aa": {"netuid": 249, "description": "ኣ ethiopic_glottal_aa"},
            "ethiopic_glottal_e": {"netuid": 250, "description": "ኤ ethiopic_glottal_e"},
            "ethiopic_glottal_ie": {"netuid": 251, "description": "እ ethiopic_glottal_ie"},
            "ethiopic_glottal_wa": {"netuid": 253, "description": "ኧ ethiopic_glottal_wa"},
            "ethiopic_wa": {"netuid": 254, "description": "ወ ethiopic_wa"},
            "ethiopic_wu": {"netuid": 255, "description": "ዉ ethiopic_wu"},
            "ethiopic_waa": {"netuid": 257, "description": "ዋ ethiopic_waa"},
            "ethiopic_ku": {"netuid": 262, "description": "኱ ethiopic_ku"},
            "ethiopic_gi": {"netuid": 270, "description": "ኒ ethiopic_gi"},
            "ethiopic_gua": {"netuid": 271, "description": "ና ethiopic_gua"},
            "ethiopic_gwe": {"netuid": 273, "description": "ን ethiopic_gwe"},
            "patrol": {"netuid": 275, "description": "अ Patrol"},
            "devanagari_aa": {"netuid": 276, "description": "आ devanagari_aa"},
            "muv": {"netuid": 277, "description": "इ muv"},
            "dogetao": {"netuid": 281, "description": "ऋ DogeTAO"}
        }
        
        # Load from config file if provided
        if config_path:
            try:
                import json
                with open(config_path, 'r') as f:
                    custom_registry = json.load(f)
                    self.registry.update(custom_registry)
            except Exception as e:
                bt.logging.warning(f"Failed to load inference registry from {config_path}: {e}")
    
    def get_netuid(self, inference_code: str) -> Optional[int]:
        """Get subnet ID for inference code"""
        if inference_code in self.registry:
            return self.registry[inference_code]["netuid"]
        return None
    
    def is_valid_code(self, inference_code: str) -> bool:
        """Check if inference code is valid"""
        return inference_code in self.registry
    
    def get_all_codes(self) -> List[str]:
        """Get all registered inference codes"""
        return list(self.registry.keys())
    
    def get_codes_for_subnet(self, netuid: int) -> List[str]:
        """Get all inference codes for a specific subnet"""
        return [code for code, info in self.registry.items() 
                if info["netuid"] == netuid]