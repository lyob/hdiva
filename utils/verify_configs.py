import os
import sys
from dataclasses import asdict

# Add the project directory to sys.path
sys.path.append("/mnt/home/blyo1/hdiva")

from b_models.configs.diva_config import (
    DiVA_ConvNet_dSprites_Training_Config as OldConvNetConfig,
)
from b_models.configs.diva_config import (
    DiVA_Manual_dSprites_Training_Config as OldManualConfig,
)
from b_models.configs.diva_config_modular import (
    DiVA_ConvNet_dSprites_Training_Config as NewConvNetConfig,
)
from b_models.configs.diva_config_modular import (
    DiVA_Manual_dSprites_Training_Config as NewManualConfig,
)


def compare_configs(old_cls, new_cls, name):
    print(f"Comparing {name}...")
    old_instance = old_cls()
    new_instance = new_cls()

    old_dict = asdict(old_instance)
    new_dict = asdict(new_instance)

    # Check for keys present in old but missing in new
    missing_keys = set(old_dict.keys()) - set(new_dict.keys())
    if missing_keys:
        print(f"FAILED: Missing keys in new config: {missing_keys}")
        return False

    # Check for keys present in new but missing in old (extra keys are okay if they are intended, but let's check)
    extra_keys = set(new_dict.keys()) - set(old_dict.keys())
    if extra_keys:
        print(f"WARNING: Extra keys in new config: {extra_keys}")

    # Compare values
    mismatch = False
    for key in old_dict:
        if key in new_dict:
            if old_dict[key] != new_dict[key]:
                print(
                    f"MISMATCH: {key} - Old: {old_dict[key]} ({type(old_dict[key])}), New: {new_dict[key]} ({type(new_dict[key])})"
                )
                mismatch = True

    if mismatch:
        print(f"FAILED: Value mismatches found in {name}")
        return False

    print(f"SUCCESS: {name} matches perfectly!")
    return True


if __name__ == "__main__":
    success = True
    success &= compare_configs(OldManualConfig, NewManualConfig, "DiVA_Manual_dSprites_Training_Config")
    print("-" * 20)
    success &= compare_configs(OldConvNetConfig, NewConvNetConfig, "DiVA_ConvNet_dSprites_Training_Config")

    if success:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)
