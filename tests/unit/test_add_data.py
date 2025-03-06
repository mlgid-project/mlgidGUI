from mlgidGUI.app.file_manager import FileManager
import shutil


def test_add_data(project_1):
    fm = FileManager()
    assert fm.root is None
    fm.open_project(project_1.root_path)
    fm.add_root_path_to_project(project_1.root_path)
    assert fm.root.folders_num == 1


def test_update_data(empty_project, project_1):

    fm = FileManager()
    fm.open_project(empty_project.root_path)
    fm.add_root_path_to_project(empty_project.root_path)
    folder_keys = list(fm.root.folder_children)
    assert len(folder_keys) == 1
    folder_key = folder_keys[0]
    assert folder_key.images_num == 0
    shutil.copyfile(project_1.root_path / '0.tiff', empty_project.root_path / '0.tiff')
    folder_key.update()
    images_num = folder_key.images_num
    assert images_num > 0

    project_path = fm.project_path
    fm.close_project()
    fm.open_project(project_path)
    folder_keys = list(fm.root.folder_children)
    assert len(folder_keys) == 1
    folder_key = folder_keys[0]
    assert folder_key.images_num == images_num