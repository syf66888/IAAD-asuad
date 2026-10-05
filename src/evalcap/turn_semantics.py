"""Classify observed turns in BDDX action/description captions."""
import re


def classify_action(action):
    s = re.sub(r"[.,;:!?]", " ", action.lower())
    s = re.sub(r"\bleftward\b", "left", s)
    s = re.sub(r"\brightward\b", "right", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Direction must relate to the observed ego action. Causal text is not action.
    s = re.split(r"\b(?:because|since)\b", s)[0].strip()
    s = re.sub(r"\bright[ -]of[ -]way\b", "priority", s)
    # A turn followed by parking on the other side is still a single turn.
    # The parking destination's side must not masquerade as a second turn.
    direction_text = s
    parking = re.search(r"\band (?:parks?|is parked|stops?) (?:on|at) the (?:left|right) side\b", s)
    if parking and re.search(r"\b(?:turn\w*|steer\w*)\b", s[:parking.start()]):
        direction_text = s[:parking.start()]
    dirs = set(re.findall(r"\b(left|right)\b", direction_text))
    direction = next(iter(dirs)) if len(dirs) == 1 else None
    out = {"category": "other", "direction": direction,
           "strict_turn": False, "broad_steering": False}
    if re.search(r"\bu[ -]?turn\b", s):
        out["category"] = "u_turn"
    elif len(dirs) > 1:
        out["category"] = "multiple_directions"
    elif (re.search(r"\b(?:steer\w*|drift\w*|pull\w*)\b.*\b(?:left|right)\b.*\band back\b", s)
          or re.search(r"\b(?:left|right)\b.*\b(?:back again|back to)\b", s)):
        out["category"] = "direction_correction_and_return"
    elif re.search(r"\b(?:wait\w*|prepar\w*|plan\w*|ready|attempt\w*|intend\w*|about)\b.*\b(?:turn\w*|steer\w*)\b", s) or re.search(r"\bbefore\s+turn(?:ing)?\b|\bto make a (?:left|right)(?: hand)? turn\b|\b(?:slow\w*|stop\w*|wait\w*)\b.*\bto turn\b", s):
        out["category"] = "planned_or_waiting_turn"
    elif re.search(r"\b(?:merg\w*|switch\w*|chang\w*)\b", s) and (direction or re.search(r"\blanes?\b", s)):
        out["category"] = "lane_change"
    elif re.search(r"\bsteer\w*\b.*\bside of (?:the|a) lane\b", s):
        out["category"] = "within_lane_adjustment"
    elif re.search(r"\b(?:mov\w*|pull\w*|shift\w*|go(?:es|ing)?|steer\w*|turn\w*|cross\w*|return\w*)\b(?:(?!\band\b).){0,50}\b(?:into|onto|to|over|one|two)\b(?:(?!\band\b).){0,40}\blanes?\b", s):
        out["category"] = "lane_change"
    elif direction and re.search(r"\b(?:curv(?:es|ed|ing)|veer(?:s|ed|ing)?)\s+(?:(?:to|toward|towards|the|its|a|little|slightly|sharply|slowly|more|very)\s+){0,5}(?:left|right)\b", s):
        out.update(category="directional_curve_or_veer", broad_steering=True)
    elif re.search(r"\b(?:curv\w*|veer\w*|swerv\w*|drift\w*|angl\w*)\b", s):
        out["category"] = "curve_or_lateral_movement"
    elif direction and (re.search(r"\bturn(?:s|ed|ing)?\s+(?:(?:to|toward|towards|the|its|a|little|slightly|sharply|slowly|haltingly|more|very)\s+){0,5}(?:left|right)\b", s) or re.search(r"\b(?:left|right)(?: hand)? turn\b", s)) and not re.search(r"\bturn lane\b", s):
        out.update(category="explicit_turn", strict_turn=True, broad_steering=True)
    elif direction and re.search(r"\b(?:make|makes|making|made)\s+(?:a |the )?(?:fast |sharp |slow )?(?:left|right)\b", s):
        out.update(category="explicit_turn", strict_turn=True, broad_steering=True)
    elif direction and re.search(r"\bsteer(?:s|ed|ing)?\s+(?:(?:to|toward|towards|the|its|a|little|slightly|sharply|slowly|more|very)\s+){0,5}(?:left|right)\b", s):
        out.update(category="steer_direction", broad_steering=True)
    elif re.search(r"\bturn(?:s|ed|ing)?\b", s) and not dirs:
        out["category"] = "turn_without_direction"
    elif dirs and re.search(r"\b(?:lane|side|shoulder|curb|park\w*|stay\w*|stick\w*|keep\w*)\b", s):
        out["category"] = "lane_position_or_keep_side"
    elif dirs:
        out["category"] = "other_directional_movement"
    return out


def classify(action):
    """Return an action category, direction, and explicit/broad turn flags."""
    raw = classify_action(action)
    kind_by_category = {
        'explicit_turn': 'turn', 'steer_direction': 'turn',
        'directional_curve_or_veer': 'turn',
        'lane_change': 'lane', 'within_lane_adjustment': 'correction',
        'direction_correction_and_return': 'correction',
        'u_turn': 'uturn', 'planned_or_waiting_turn': 'planned',
        'multiple_directions': 'ambiguous',
        'curve_or_lateral_movement': 'ambiguous',
        'turn_without_direction': 'ambiguous',
        'lane_position_or_keep_side': 'position',
        'other_directional_movement': 'ambiguous', 'other': 'none',
    }
    return dict(kind=kind_by_category[raw['category']], **raw)
