"""
This script contains functions for character augmentation based on keyboard distances.
"""

from typing import Dict, Set, Tuple
from functools import partial

from pydantic import BaseModel

from .character import char_replace_augmenter

qwerty_en_array = {"default": [
    ['`','1','2','3','4','5','6','7','8','9','0','-','='],
    ['q','w','e','r','t','y','u','i','o','p','[',']','\\'],
    ['a','s','d','f','g','h','j','k','l',';','\''],
    ['z','x','c','v','b','n','m',',','.','/'],
    ],
    "shifted": [
    ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    ]}


qwerty_da_array = {"default": [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '´'],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å', '¨'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'æ', 'ø', "'"], 
    ['<', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '-']],
    "shifted": [
    ['!', '"', '#', '€', '%', '&', '/', '(', ')', '=', '?', '`'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'Å', '^'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Æ', 'Ø', '*']
    ['>', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ';', ':', '_']
    ]}
KEYBOARDS = {'QWERTY_EN': qwerty_en_array,
'QWERTY_DA': qwerty_da_array,
}

class Keyboard(BaseModel):
    keyboard_array = Dict[str, str]
    shift_distance = 3

    def coordinate(self, key: str) -> Tuple[int, int]:
        for arr in self.keyboard_array:
            for x, row in enumerate(self.keyboard_array[arr]):
                for y, k in enumerate(row):
                    if key == k:
                        return x, y

        raise ValueError(f"key {key} was not found in keyboard array")

    def is_shifted(self, key: str) -> bool:
        for x in self.keyboard_array["shifted"]:
            if key in x:
                return True
        return False
        

    def euclidian_distance(self, key_a: str, key_b: str) -> int:
        x1, y1 = self.coordinate(key_a)
        x2, y2 = self.coordinate(key_b)

        shift_cost = 0 if self.is_shifted(key_a) == self.is_shifted(key_b) else self.shift_distance

        return ((x1 - x2)**2 + (y1-y2)**2)**0.5 + shift_cost

    def all_keys(self):
        for arr in self.keyboard_array: 
            for x, _ in enumerate(self.keyboard_array[arr]):
                for k in self.keyboard_array[arr][x]:
                    yield k

    def get_neighboors(self, key: str, distance: int=1) -> Set[int]:
        l = []
        for k in self.all_keys():
            if k == key:
                continue
            if self.euclidian_distance(key, k) <= distance:
                l.append(k)
        return l

    def create_distance_dict(self, distance: int=1) -> dict:
        return {k: self.get_neighboors(k, distance=distance)for k in self.all_keys()}


def make_keyboard_augmenter(char_level: float, doc_level: float, distance=1, keyboard: str = "QWERTY_EN"):
    kb = KEYBOARDS[keyboard]
    replace_dict = kb.create_distance_dict(distance=distance)
    return partial(char_replace_augmenter, replacement = replace_dict, doc_level = doc_level, char_level = char_level)
