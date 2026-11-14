PLAYER_CLASS_ID = 2
HEAD_CLASS_ID = 1

HIDEOUT_TARGET_CLASS_IDS = (5, 6)

CLASS_NAMES = {
    0: "flash",
    HEAD_CLASS_ID: "head",
    PLAYER_CLASS_ID: "player",
    3: "smoke",
}

NON_TARGET_CLASS_IDS = {0, 3}


def is_head_class(class_id) -> bool:
    try:
        return int(class_id) == HEAD_CLASS_ID
    except (TypeError, ValueError):
        return False
