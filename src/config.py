from dataclasses import dataclass


@dataclass
class JigsawConfig:
    d_image: tuple[int, int] = (32, 32)
    n_patches: int = 4
    n_tiles: int = 16
    d_patch: tuple[int, int] = (8, 8)


cfg = JigsawConfig()
