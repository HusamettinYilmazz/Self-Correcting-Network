from models.self_correcting_model.self_correction_interface import SelfCorrectionModule
from models.self_correcting_model.self_correction_enums import SelfCorrectingEnums

from .networks import NoCorrectionModule, LinearCorrectionModule, ConvCorrectionModule

class SelfCorrectingNetwrokFactory:
    def __init__(self):
        ...

    def build_correction_module(self, variant: str, num_classes: int, 
        alpha_start: float = 30.0, alpha_end: float = 0.5, ) -> SelfCorrectionModule:
        
        
        variant = variant.lower()
        if variant == SelfCorrectingEnums.NO_CORRECTION.value:
            return NoCorrectionModule()
        elif variant == SelfCorrectingEnums.Linear_CORRECTION.value:
            return LinearCorrectionModule(alpha_start=alpha_start, alpha_end=alpha_end)
        elif variant == SelfCorrectingEnums.CONV_CORRECTION.value:
            return ConvCorrectionModule(num_classes=num_classes)
        
        else:
            raise ValueError(
                f"Unknown correction variant '{variant}'. "
                f"Choose from: 'no_correction', 'linear_correction', 'conv_correction'"
            )