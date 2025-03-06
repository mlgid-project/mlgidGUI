from mlgidGUI import App


def test_loading_state(project_1):
    App.clear()
    App.__new__(App)
    app = App(project_1.root_path)
    app.fm.open_project(project_1.root_path)
    app.fm.add_root_path_to_project(project_1.root_path)
    image_keys = list(project_1.all_image_keys())
    image_key_with_segments = image_keys[0]
    app.fm.change_image(image_key_with_segments)

    app.roi_dict.create_roi(radius=1, width=2)
    app.roi_dict.create_roi(radius=3, width=3)
    app.save_state()
    assert len(app.roi_dict) == 2

    app.close()
    app = App(project_1.root_path)
    assert len(app.roi_dict) == 2