from dacy.augmenters.keyboard import Keyboard, qwerty_da_array
from dacy.augmenters import create_keyboard_augmenter

def test_Keyboard():
    kb = Keyboard(keyboard_array = qwerty_da_array)

    assert kb.coordinate("q") == (1, 1)
    assert kb.is_shifted()("q") is False
    assert kb.euclidian_distance("q", "a") <= 1
    assert len(set(kb.all_keys())) > 28*2
    assert "w" in kb.get_neighboors("q")
    kb.create_distance_dict()

def test_make_keyboard_augmenter():
    augmenter = create_keyboard_augmenter(doc_level=1, char_level=0.5)