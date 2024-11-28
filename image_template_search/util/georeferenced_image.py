import typing
# Define enumerations based on the dictionary values
from enum import Enum
from typing import Tuple
import datetime
from pydantic import BaseModel
from pydantic import Field


class ColorSpace(Enum):
    SRGB = 1
    UNCALIBRATED = 65535


class ExposureMode(Enum):
    AUTO_EXPOSURE = 0
    MANUAL_EXPOSURE = 1
    AUTO_BRACKET = 2


class ExposureProgram(Enum):
    NOT_DEFINED = 0
    MANUAL = 1
    NORMAL_PROGRAM = 2
    APERTURE_PRIORITY = 3
    SHUTTER_PRIORITY = 4
    CREATIVE_PROGRAM = 5  # For depth of field
    ACTION_PROGRAM = 6    # For motion capture
    PORTRAIT_MODE = 7     # For close-up photos with the background out of focus
    LANDSCAPE_MODE = 8    # For landscape photos with the background in focus


class GpsAltitudeRef(Enum):
    ABOVE_SEA_LEVEL = 0
    BELOW_SEA_LEVEL = 1


class LightSource(Enum):
    UNKNOWN = 0
    DAYLIGHT = 1
    FLUORESCENT = 2
    TUNGSTEN = 3
    FLASH = 4
    FINE_WEATHER = 9
    CLOUDY_WEATHER = 10
    SHADE = 11
    DAYLIGHT_FLUORESCENT = 12
    DAY_WHITE_FLUORESCENT = 13
    COOL_WHITE_FLUORESCENT = 14
    WHITE_FLUORESCENT = 15
    STANDARD_LIGHT_A = 17
    STANDARD_LIGHT_B = 18
    STANDARD_LIGHT_C = 19
    D55 = 20
    D65 = 21
    D75 = 22
    D50 = 23
    ISO_STUDIO_TUNGSTEN = 24
    OTHER = 255


class MeteringMode(Enum):
    UNKNOWN = 0
    AVERAGE = 1
    CENTER_WEIGHTED_AVERAGE = 2
    SPOT = 3
    MULTISPOT = 4
    PATTERN = 5
    PARTIAL = 6
    OTHER = 255


class Orientation(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4
    LEFT_TOP = 5
    RIGHT_TOP = 6
    RIGHT_BOTTOM = 7
    LEFT_BOTTOM = 8


class ResolutionUnit(Enum):
    NONE = 1
    INCHES = 2
    CENTIMETERS = 3


class Saturation(Enum):
    NORMAL = 0
    LOW = 1
    HIGH = 2


class SceneCaptureType(Enum):
    STANDARD = 0
    LANDSCAPE = 1
    PORTRAIT = 2
    NIGHT_SCENE = 3


class Sharpness(Enum):
    NORMAL = 0
    SOFT = 1
    HARD = 2


class WhiteBalance(Enum):
    AUTO = 0
    MANUAL = 1


class ExifMetaData(BaseModel):
    """
    retrieve exif metadata from a georeferenced mavic 2 pro image
    """
    _exif_ifd_pointer: int
    _gps_ifd_pointer: int
    bits_per_sample: int
    body_serial_number: str
    color_space: ColorSpace
    compression: int
    contrast: int
    datetime: str  # Original format appears to be 'YYYY:MM:DD HH:MM:SS'
    datetime_digitized: datetime.datetime
    datetime_original: str  # Same format as `datetime`
    digital_zoom_ratio: float
    exif_version: str
    exposure_bias_value: float
    exposure_mode: ExposureMode
    exposure_program: ExposureProgram
    exposure_time: float
    f_number: float
    filepath: str
    focal_length: float
    focal_length_in_35mm_film: int
    gain_control: int
    gps_altitude: float
    gps_altitude_ref: GpsAltitudeRef
    gps_latitude: Tuple[float, float, float]
    gps_latitude_ref: str
    gps_longitude: Tuple[float, float, float]
    gps_longitude_ref: str
    gps_version_id: int
    image_description: str
    image_height: int
    image_width: int
    jpeg_interchange_format: int
    jpeg_interchange_format_length: int
    latitude: float
    lens_specification: Tuple[float, float, float, float]
    light_source: LightSource
    longitude: float
    make: str
    max_aperture_value: float
    metering_mode: MeteringMode
    model: str
    orientation: Orientation
    photographic_sensitivity: int
    pixel_x_dimension: int
    pixel_y_dimension: int
    resolution_unit: ResolutionUnit
    samples_per_pixel: int
    saturation: Saturation
    scene_capture_type: SceneCaptureType
    sharpness: Sharpness
    software: str
    white_balance: WhiteBalance
    x_resolution: float
    xp_keywords: str
    y_and_c_positioning: int
    y_resolution: float

    class Config:
        use_enum_values = True


class ExtendImageMetaData(ExifMetaData):
    image_id: typing.Union[str, int] = Field(alias='image_id')
    image_name: str = Field(alias='image_name', description="Name of the image file")

class XMPMetaData(BaseModel):
    """
    retrieve xmp metadata from a georeferenced mavic 2 pro image
    """