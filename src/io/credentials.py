"""Shared runtime credential loader for migrated agents.

Based on the hardened pattern used by notebooks 07/08: resolve once into the
process environment and replace ``google.colab.userdata`` with a runtime-only
proxy. Compatibility with the credential file materialized by the current
notebook 00 (``.runtime_secrets/openai_api_key.txt``, plain text, via the
``llm_utils.py`` it writes) is preserved. The current notebook 00 does not
materialize a Fernet-encrypted ``.secrets/openai_api_key.key``/``.enc`` pair;
that legacy mechanism was removed from this module in
CREDENTIALS-RESIDUAL-02.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any, Callable, Mapping, MutableMapping


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("OPENAI_API_KEY", "api_key", "key", "value"):
            if key in value:
                return str(value[key] or "").strip()
        return str(next(iter(value.values()), "") or "").strip()
    return str(value).strip()


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"_project_llm_utils_{abs(hash(path.resolve()))}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar PROJECT_DIR/src/llm_utils.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_from_project_llm_utils(
    project_dir: Path,
    *,
    loader: Callable[[Path], Any] | None = None,
) -> str | None:
    llm_utils_path = project_dir / "src" / "llm_utils.py"
    if not llm_utils_path.is_file():
        return None
    try:
        module = (loader or _load_module)(llm_utils_path)
    except Exception:
        raise RuntimeError(
            "No se pudo cargar el mecanismo de credenciales del notebook 00."
        ) from None

    runtime_secret_dir = project_dir / ".runtime_secrets"
    for name, value in {
        "PROJECT_DIR": project_dir,
        "RUNTIME_SECRET_DIR": runtime_secret_dir,
        "OPENAI_KEY_FILE": runtime_secret_dir / "openai_api_key.txt",
    }.items():
        setattr(module, name, value)

    ensure_key = getattr(module, "ensure_openai_key", None)
    try:
        if callable(ensure_key):
            value = _normalize(
                ensure_key(allow_prompt=False, persist_if_prompted=False)
            )
            return value or None
    except Exception:
        return None
    return None


def install_runtime_userdata_proxy(
    *,
    project_dir: str | Path,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    environment = os.environ if environ is None else environ
    root = Path(project_dir).resolve()

    class RuntimeOnlyUserdata(types.ModuleType):
        def get(self, key: str, *args: Any, **kwargs: Any) -> str:
            return load_runtime_credential(
                str(key), project_dir=root, environ=environment, required=True
            )

    proxy = RuntimeOnlyUserdata("google.colab.userdata")
    sys.modules["google.colab.userdata"] = proxy
    try:
        import google.colab as google_colab
        setattr(google_colab, "userdata", proxy)
    except Exception:
        pass


def load_runtime_credential(
    secret_name: str,
    *,
    project_dir: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
    colab_userdata_getter: Callable[[str], Any] | None = None,
    llm_utils_loader: Callable[[Path], Any] | None = None,
    required: bool = True,
) -> str | None:
    """Resolve one credential without logging, fingerprinting, or prompting."""
    name = str(secret_name).strip()
    if not name:
        raise ValueError("secret_name no puede estar vacío.")
    environment = os.environ if environ is None else environ
    root = Path(
        project_dir
        if project_dir is not None
        else environment.get("THESIS_PROJECT_DIR", "/content/proyecto_estado_arte")
    ).resolve()

    def cache(value: str) -> str:
        environment[name] = value
        if name == "OPENAI_API_KEY":
            install_runtime_userdata_proxy(
                project_dir=root,
                environ=environment,
            )
        return value

    value = _normalize(environment.get(name, ""))
    if value:
        return cache(value)

    if name == "OPENAI_API_KEY":
        value = _load_from_project_llm_utils(root, loader=llm_utils_loader)
        runtime_file = root / ".runtime_secrets" / "openai_api_key.txt"
        if not value and runtime_file.is_file():
            value = runtime_file.read_text(encoding="utf-8").strip() or None
        if value:
            return cache(value)

    getter = colab_userdata_getter
    if getter is None:
        try:
            from google.colab import userdata
        except Exception:
            getter = None
        else:
            getter = userdata.get
    if getter is not None:
        try:
            value = _normalize(getter(name))
        except Exception:
            value = ""
    if value:
        return cache(value)

    if required:
        raise RuntimeError(
            f"La credencial {name!r} no está disponible. Ejecuta primero el notebook 00."
        )
    return None


def resolve_openai_api_key(
    *,
    project_dir: str | Path | None = None,
    required: bool = True,
    environ: MutableMapping[str, str] | None = None,
    colab_userdata_getter: Callable[[str], Any] | None = None,
    llm_utils_loader: Callable[[Path], Any] | None = None,
) -> str | None:
    return load_runtime_credential(
        "OPENAI_API_KEY",
        project_dir=project_dir,
        required=required,
        environ=environ,
        colab_userdata_getter=colab_userdata_getter,
        llm_utils_loader=llm_utils_loader,
    )
