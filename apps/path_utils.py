import os


def to_wsl_unc_path(path: str, distro_name: str = "Ubuntu-24.04") -> str:
    if not path:
        return path

    # Avoid double conversion when the input is already a UNC path.
    unc_prefix = "\\\\wsl.localhost\\"
    if path.lower().startswith(unc_prefix):
        return os.path.normpath(path)

    abs_path = os.path.abspath(path)
    linux_like = abs_path.replace("\\", "/")
    if linux_like.startswith("/"):
        return "\\\\wsl.localhost\\{}{}".format(distro_name, linux_like.replace("/", "\\"))
    return abs_path


def normalize_config_path(path: str) -> str:
    candidate = path.strip().strip('"').strip("'")
    if not candidate:
        return candidate

    is_windows = os.name == "nt"

    if candidate.startswith("\\\\wsl.localhost\\"):
        if is_windows:
            return os.path.normpath(candidate)
        parts = candidate.split("\\")
        if len(parts) >= 5:
            return "/" + "/".join(parts[4:])

    if len(candidate) >= 2 and candidate[1] == ":":
        if is_windows:
            return os.path.abspath(os.path.expanduser(candidate))
        drive_letter = candidate[0].lower()
        rest = candidate[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive_letter}{rest}"

    return os.path.abspath(os.path.expanduser(candidate))


def sanitize_for_dirname(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "default"
