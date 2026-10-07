"""pygame keyboard module (kept for the calibration scripts).

The flight loop uses ui.unified_gui instead; calibration scripts still
use init() + getKeyPressedOnce() for their emergency-abort checks.
"""

import pygame

# FIX: track previous key states so we can detect a fresh "press"
# (transition from not-pressed to pressed) instead of "held down".
# This is what stops takeoff/land from firing over and over while
# a key is held, since getKey() alone reports true every single loop.
_prev_key_state = {}


def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))


def getKey(keyName):
    """Returns True every loop iteration the key is held down (original behavior)."""
    ans = False
    for eve in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))

    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


def getKeyPressedOnce(keyName):
    """
    Returns True only on the frame the key transitions from up -> down.
    Use this for actions that must fire once per press (takeoff, land,
    flip, mode toggle) instead of getKey(), which fires every loop
    while the key is held.
    """
    if not pygame.display.get_init():
        # pygame was never initialised (e.g. the Tk flight GUI is in use):
        # it cannot see any keys, so report "not pressed" instead of
        # raising "video system not initialized".
        return False
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    is_down = bool(keyInput[myKey])

    was_down = _prev_key_state.get(keyName, False)
    _prev_key_state[keyName] = is_down

    return is_down and not was_down


def main():
    if getKey("LEFT"):
        pass

    if getKey("RIGHT"):
        pass


if __name__ == "__main__":
    init()
    while True:
        main()
