from ..rois.roi import Roi, RoiTypes


def _update_bounds(roi: Roi, param_names, init_params, upper_bounds, lower_bounds):
    if roi.fitted_parameters:
        try:
            new_init_params = [roi.fitted_parameters[k] for k in param_names]
            for j, (l, i, u) in enumerate(zip(lower_bounds, new_init_params, upper_bounds)):
                if l <= i <= u:
                    init_params[j] = i
        except KeyError:
            pass
    return init_params, upper_bounds, lower_bounds


def _get_dummy_bounds(num: int):
    return [0.5] * num, [1] * num, [0] * num
