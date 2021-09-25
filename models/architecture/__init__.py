from .base_architecture import BaseArchitecture
from .top_down_architecture import TopDownArchitecture
from .bottom_up_architecture import BottomUpArchitecture

def get_architecture_class(architecture: str) -> BaseArchitecture:
    if architecture == 'top-down':
        return TopDownArchitecture
    elif architecture == 'bottom-up':
        return BottomUpArchitecture
    else:
        raise NotImplementedError(f'Architecture {architecture} is not implemented')