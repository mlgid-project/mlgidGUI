from mlgidGUI import App


def test_create_rois(project_1):
    image_keys = list(project_1.all_image_keys())
    App.clear()
    App.__new__(App)
    app = App(project_1.root_path)
    app.fm.open_project(project_1.root_path)
    assert image_keys[0] is not None
    app.fm.change_image(image_keys[0])

    assert app.roi_dict._current_key is not None

    assert len(app.roi_dict) == 0
    app.roi_dict.create_roi(radius=1, width=2)
    app.roi_dict.create_roi(radius=3, width=3)
    assert len(app.roi_dict) == 2