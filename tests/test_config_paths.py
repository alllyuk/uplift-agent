import importlib
import os


def test_paths_config_artifacts_dir_created(tmp_path):
    # Override artifacts dir via env before importing the module
    target = tmp_path / "artifacts_custom"
    os.environ["PATHS_ARTIFACTS_DIR"] = str(target)

    # Import fresh to apply env overrides
    cfg_mod = importlib.import_module("sme_causal.core.config")
    importlib.reload(cfg_mod)

    app_cfg = cfg_mod.AppConfig()
    # Artifacts dir should be created by validator
    assert target.exists() and target.is_dir()
    # Derived paths should resolve under artifacts dir
    assert str(app_cfg.synthetic_clients_path).startswith(str(target))
    assert str(app_cfg.pipeline_log_path).startswith(str(target))

