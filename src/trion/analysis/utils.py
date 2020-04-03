
def is_optical_signal(role: str) -> bool:
    return role.startswith("sig")

def is_tap_modulation(role: str) -> bool:
    return role.startswith("tap")

def is_pshet_modulation(role: str) -> bool:
    return role.startswith("ref")
