# -*- coding: utf-8 -*-

from .basic_profile import BasicProfile


class AngularProfile(BasicProfile):

    def __init__(self, image_holder, parent=None):
        self.image_holder = image_holder
        super().__init__(parent)

    def update_data_from_source(self, key: int = None):

        image_holder = self.image_holder
        profile = image_holder.get_angular_profile(key)
        if profile is None:
            return
        phi_axis = image_holder.geometry.phi_axis
        if phi_axis is not None and phi_axis.size == profile.size:
            self.set_data(profile, phi_axis)
