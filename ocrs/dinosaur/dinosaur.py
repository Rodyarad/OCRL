from ocrs.base import Base

from .dinosaur_module import Dinosaur_Module


class Dinosaur(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = Dinosaur_Module(ocr_config, env_config)
        super(Dinosaur, self).__init__(ocr_config, env_config)
