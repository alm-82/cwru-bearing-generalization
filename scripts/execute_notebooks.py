"""Execute safe notebook defaults through Jupyter, with bounded child processes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import signal
import asyncio

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/notebook_execution"


def execute_one(path=None):
    # On Windows, ZeroMQ needs selector callbacks. The Proactor compatibility
    # thread completed cells here but exited with STATUS_INVALID_HANDLE at teardown.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("Importing notebook client", flush=True)
    import nbformat
    from nbclient import NotebookClient
    notebook = (nbformat.read(path, as_version=4) if path else nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("value = sum([1, 2, 3])\nassert value == 6\nvalue")]))
    client = NotebookClient(notebook, timeout=60, startup_timeout=60, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    print("Executing through Jupyter", flush=True)
    client.execute()
    if path is None:
        assert notebook.cells[0].outputs[0]["data"]["text/plain"] == "6"
    nbformat.write(notebook, WORK / (path.name if path else "minimal_executed.ipynb"))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("CWRU_DATA_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    for key, directory in [("JUPYTER_CONFIG_DIR", "config"), ("JUPYTER_RUNTIME_DIR", "runtime"),
                           ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib"),
                           ("TEMP", "temp"), ("TMP", "temp")]:
        target = WORK / directory
        target.mkdir(exist_ok=True)
        env[key] = str(target)
    results = []
    for path in [None, *sorted((ROOT / "notebooks").glob("*.ipynb"))]:
        name = path.name if path else "minimal_preflight"
        command = [sys.executable, "-u", "-B", __file__, "--one", str(path)] if path else [sys.executable, "-u", "-B", __file__, "--minimal"]
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, **options)
        try:
            stdout, stderr = child.communicate(timeout=180)
            row = {"notebook": name, "passed": child.returncode == 0,
                   "stdout": stdout, "stderr": stderr}
        except subprocess.TimeoutExpired as timeout:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(child.pid, signal.SIGKILL)
            try:
                stdout, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired as cleanup_timeout:
                child.kill()
                # A descendant may retain pipe handles after the parent exits.
                # Preserve available diagnostics without blocking the whole runner.
                stdout = cleanup_timeout.output or timeout.output or b""
                stderr = cleanup_timeout.stderr or timeout.stderr or b""
                child.stdout.close()
                child.stderr.close()
                stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
            row = {"notebook": name, "passed": False, "error": "180-second process limit",
                   "stdout": stdout, "stderr": stderr}
        results.append(row)
        print(json.dumps(row), flush=True)
        (WORK / "execution.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if path is None and not row["passed"]:
            break  # Diagnose startup once instead of repeating it for every notebook.
    (WORK / "execution.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        execute_one()
    elif len(sys.argv) > 1 and sys.argv[1] == "--one":
        execute_one(Path(sys.argv[2]))
    else:
        raise SystemExit(main())
