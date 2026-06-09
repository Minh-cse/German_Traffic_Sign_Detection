import os
import sys
import shutil
import subprocess

here = os.path.abspath(os.path.dirname(__file__))
backend_cwd = os.path.join(here, "backend")
frontend_cwd = os.path.join(here, "Frontend", "Traffic-sign-recognition")

backend_cmd = [sys.executable, "-m", "uvicorn", "app:app", "--reload"]

npm_exe = shutil.which("npm") or shutil.which("npm.cmd")
if not npm_exe:
    print("Error: 'npm' not found. Install Node.js or add npm to PATH. Run 'where npm' to check.")
    sys.exit(1)

frontend_cmd = [npm_exe, "run", "dev"]

if not os.path.isdir(backend_cwd):
    print(f"Error: backend cwd not found: {backend_cwd}")
    sys.exit(1)
if not os.path.isdir(frontend_cwd):
    print(f"Error: frontend cwd not found: {frontend_cwd}")
    sys.exit(1)

backend = subprocess.Popen(backend_cmd, cwd=backend_cwd)
frontend = subprocess.Popen(frontend_cmd, cwd=frontend_cwd, env=os.environ.copy())

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()
