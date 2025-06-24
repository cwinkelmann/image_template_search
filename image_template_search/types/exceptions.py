##
class NoMatchError(ValueError):
    """Raised when no matches are found in the image template search."""


class DetailedNoMatchError(ValueError):
    """Raised when no matches are found in the image template search."""

    def __init__(self, message, template_path, large_image_path, *args, **kwargs):
        super().__init__(message, *args)

        self.template_path = template_path
        self.large_image_path = large_image_path