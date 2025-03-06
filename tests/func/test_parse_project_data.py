from mlgidGUI import App


def test_project_1_images(project_1):
    """Project 1 should have a certain data"""
    App.clear()
    App.__new__(App)
    app = App(project_1.root_path)
    app.fm.open_project(project_1.root_path)


    assert app.fm.root.images_num == 0
    assert app.fm.root.folders_num == 0

    for image in project_1.all_image_keys():
        app.fm.add_root_path_to_project(image.path)

    assert app.fm.root.images_num == 3

    for folder in project_1.root_key().folder_children:
        app.fm.add_root_path_to_project(folder.path)
    assert app.fm.root.images_num == 3
    assert app.fm.root.folders_num == 6