import logging

from .rois.roi_dict import RoiDict
from .file_manager import FileManager, ImageKey
from .utils import SingletonMeta
from .geometry import Geometry
from .geometry_holder import GeometryHolder
from .image_holder import ImageHolder
from .profiles import RadialProfile, AngularProfile
from .debug_tracker import TrackQObjects
from .data_manager import DataManager


class App(metaclass=SingletonMeta):
    log = logging.getLogger(__name__)

    def __init__(self, parent, *, _connect: bool = True):
        self.parent = parent
        self.fm: FileManager = FileManager()
        self.geometry_holder: GeometryHolder = GeometryHolder(self.fm)
        self.roi_dict: RoiDict = RoiDict(self.fm, self.geometry_holder)
        self.image_holder: ImageHolder = ImageHolder(self.fm, self.geometry_holder, self.roi_dict)
        self.radial_profile: RadialProfile = RadialProfile(self.image_holder, self.fm)
        self.angular_profile: AngularProfile = AngularProfile(self.image_holder)
        self.data_manager: DataManager = DataManager(self.fm, self.image_holder)

        if _connect:
            self._connect_app()

        self.debug_tracker: TrackQObjects or None = None
        if self.log.level <= logging.DEBUG:
            self.debug_tracker = TrackQObjects()

    def save_state(self):
        self.geometry_holder.save_state()
        self.roi_dict.save_state()
        # self.radial_profile.save_state()

    @property
    def image(self):
        return self.image_holder.image

    @property
    def polar_image(self):
        return self.image_holder.polar_image

    @property
    def geometry(self) -> Geometry:
        return self.geometry_holder.geometry

    @property
    def current_image_key(self) -> ImageKey or None:
        return self.fm.current_key

    def close(self):
        self.fm.close()

    def _connect_app(self):
        self.image_holder.sigPolarImageChanged.connect(self.radial_profile.update)
        self.image_holder.sigPolarImageChanged.connect(self.angular_profile.update)

        self.geometry_holder.sigScaleChanged.connect(self.roi_dict.on_scale_changed)
        self.geometry_holder.sigRingBoundsChanged.connect(self.roi_dict.change_ring_bounds)

        self.fm.sigActiveImageChanged.connect(self.image_holder.change_image)
        self.fm.sigProjectIsClosing.connect(self.save_state)
        self.fm.sigActiveFolderChanged.connect(self.roi_dict.change_folder)
