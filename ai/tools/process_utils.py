"""Bound execution time, including Windows venv launcher descendants."""
import subprocess


def run_with_timeout(command, timeout, cwd=None):
    import psutil

    process = subprocess.Popen(command, cwd=cwd)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
        except psutil.NoSuchProcess:
            pass
        process.wait()
        raise
