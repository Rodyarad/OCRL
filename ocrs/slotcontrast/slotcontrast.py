from ocrs.base import Base

from .slotcontrast_module import SlotContrast_Module


class SlotContrast(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = SlotContrast_Module(ocr_config, env_config)
        super(SlotContrast, self).__init__(ocr_config, env_config)
