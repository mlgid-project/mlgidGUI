from pathlib import Path
from mlgidGUI import App


def convert_project_to_h5(project_dir: Path, h5_file: Path, skip_unlabelled_images: bool = True):
    app = App(_connect=False)
    project_dir = Path(project_dir)
    h5_file = Path(h5_file)

    app.fm.open_project(project_dir)
    app.data_manager.project2h5(h5_file, skip_empty_images=skip_unlabelled_images)
