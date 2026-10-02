from pygame import SRCALPHA, Color, Surface
from pygame.draw import circle
from pygame.sprite import Sprite


class TouchTargetSprite(Sprite):
    """Concentric-circle touch target, ported from
    gongo-frequency-discriminate/assets/target_sprite.py since this project
    has no equivalent built-in asset."""

    def __init__(self, size: int, center: tuple[int, int]) -> None:
        super().__init__()
        self.image = Surface((size, size), SRCALPHA)
        self.image.fill(Color("lightgray"))
        for x_ratio, y_ratio, radius_divisor, color in (
            (0.5, 0.5, 2.1, "#616161"),
            (0.5, 0.5, 3.1, "#bababa"),
            (0.5, 0.5, 6.3, "white"),
        ):
            circle(
                self.image,
                Color(color),
                (round(size * x_ratio), round(size * y_ratio)),
                round(size / radius_divisor),
            )
        self.rect = self.image.get_rect(center=center)
