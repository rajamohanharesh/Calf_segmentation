#From https://github.com/yiskw713/asrf

import dataclasses
import pprint
from typing import Any, Dict, Tuple

import yaml

__all__ = ["get_config"]




@dataclasses.dataclass
class Config:
    OBJECTIVE: str = "muscles"
    NUM_CLASSES: int = 6
    NUM_CHANNELS: int = 1
    MODEL: str = "unet"
    BATCH_SIZE: int = 8
    FOLD: int = 1
    LEARNING_RATE: float = 1e-5
    NUM_EPOCHS: int = 400
    DROPOUT_RATE: float = 0.98
    beta_dice: float = 1
    beta_youden: float = 0
    beta_ce: float = 0
    scaling: bool = False
    rotation: bool = True
    cropping: bool = True
    normalize: bool = True
    flipping: bool = True

    


    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) Tuple[int] -> tuple
            # https://stackoverflow.com/questions/51171908/extracting-data-from-typing-types

            # check the instance is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )

    def _type_check_element(
        self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )


def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use list in dataclass because mutable defaults are not allowed.
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config
