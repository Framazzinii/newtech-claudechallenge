# Setting up the ABIDES lab on Windows

This is the step-by-step setup guide for Windows 10 and Windows 11. It assumes
you have never installed Docker before. Allow about 30–45 minutes for the
first run, mostly waiting on downloads.

If you are on macOS, see [SETUP_MACOS.md](SETUP_MACOS.md) instead.

## What you'll install

1. WSL2 (Windows Subsystem for Linux 2) — already on most Windows 11
   machines.
2. Docker Desktop — runs the lab container.
3. Git for Windows — clones the repo and gives you Git Bash, a Unix-like
   shell.
4. (Recommended) GNU make — lets you use the short `make build`, `make smoke`
   commands. You can skip this and run the longer `docker compose ...`
   commands instead; both are documented below.
5. (Optional) Visual Studio Code with the *Dev Containers* extension —
   one-click reopen-in-container experience.

You do **not** need to install Python, NumPy, JupyterLab, or any of ABIDES's
dependencies on your host machine. Everything lives inside the container.

---

## Step 1 — Confirm hardware virtualisation is enabled

Docker Desktop on Windows runs the Linux container inside a lightweight VM
(via WSL2). That requires hardware virtualisation (Intel VT-x or AMD-V),
which is sometimes disabled in the BIOS/UEFI on factory laptops.

**Check whether it's enabled:**

1. Press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Esc</kbd> to open Task Manager.
2. Click the **Performance** tab, then **CPU**.
3. Look at the bottom right for **Virtualization**. You want it to say
   **Enabled**.

**If it says Disabled:**

1. Reboot your laptop and enter the BIOS/UEFI (the key is shown briefly at
   boot — typically <kbd>F2</kbd>, <kbd>F10</kbd>, <kbd>Del</kbd>, or
   <kbd>Esc</kbd>).
2. Look for a setting called *Intel Virtualization Technology*,
   *Intel VT-x*, *SVM Mode* (AMD), or *Hyper-V*. Enable it.
3. Save and reboot. Re-check Task Manager.

If you cannot find the setting or BIOS access is locked by your
university's IT, talk to the teaching team about a GitHub Codespaces fallback
— Codespaces runs the container in the cloud and only needs a browser.

---

## Step 2 — Install WSL2

This is what Docker Desktop actually uses to run Linux containers. The
installer runs Docker on top of it.

Open **PowerShell as Administrator** (right-click the Start menu → *Terminal
(Admin)* on Windows 11, or *Windows PowerShell (Admin)* on Windows 10) and
run:

```powershell
wsl --install
```

This installs WSL2 and a default Ubuntu distribution. Reboot when prompted.
After reboot, an Ubuntu window may pop up asking for a username and password
— set them to whatever you like; the lab does not use this Ubuntu directly,
but having it installed is required for Docker Desktop's WSL2 backend.

If you already have WSL2 installed, `wsl --install` will tell you so. That's
fine, skip ahead.

**Verify:**

```powershell
wsl --status
```

You want to see `Default Version: 2`. If it says version 1, run
`wsl --set-default-version 2`.

---

## Step 3 — Install Docker Desktop

1. Download Docker Desktop for Windows from
   <https://www.docker.com/products/docker-desktop/>.
2. Run the installer. When asked, **leave the *Use WSL 2 instead of Hyper-V*
   option ticked** (it's the default).
3. Reboot when the installer asks.
4. Launch Docker Desktop from the Start menu. If you get an error message complaining that
   "Docker Desktop must be owned by an elevated account", then try the following workaround: 
   run PowerShell as admininistrator, then use it to delete and recreate the DockerDesktop folder by typing

```powershell
rmdir C:\ProgramData\DockerDesktop
cd C:\ProgramData
mkdir DockerDesktop
```
   If after this you still cannot launch Docker Desktop, 
   ask the teaching team for assistance.
5. Accept the licence agreement (it's free for personal/educational use).
6. Wait until the whale icon in the system tray is steady (not animated).
   This means the Docker engine is running. The first start can take a
   minute or two.

**Verify** (use any terminal — PowerShell, Command Prompt, or Git Bash):

```powershell
docker --version
docker run --rm hello-world
```

The second command pulls a tiny test image and prints "Hello from Docker!".
If both work, Docker is healthy.

If `docker run hello-world` fails with something about WSL2 or virtualisation,
go back to Steps 1 and 2.

---

## Step 4 — Install Git

Download **Git for Windows** from <https://gitforwindows.org/> and run the
installer. Accept the defaults; the only choice that matters for this lab is
**Checkout as-is, commit Unix-style line endings** when the installer asks
about line endings — the repo's `.gitattributes` file enforces LF endings on
shell scripts and the Dockerfile, so this option keeps Git from second-
guessing.

The installer also gives you **Git Bash**, a Unix-like terminal. We recommend
using Git Bash for the rest of this guide because the commands match what
Mac and Linux students see, and a few `make` recipes assume bash semantics.

**Verify** in Git Bash:

```bash
git --version
```

---

## Step 5 — Terminal orientation

It's worth noting that you have several terminals available:

| Terminal              | Verdict for this lab                                                |
| --------------------- | ------------------------------------------------------------------- |
| **Git Bash**          | Recommended. Behaves like Mac/Linux; `make` works once installed.   |
| PowerShell            | Works and may be needed for certain side activities. `make` needs to be installed separately.             |
| Command Prompt        | Avoid — quoting and path semantics differ from the recipes.         |
| WSL2 Ubuntu shell     | Overkill and confusing — Docker Desktop uses WSL2 internally, but you don't need to drop into it for this lab. |

For the rest of this guide, commands shown are for **Git Bash**.

---

## Step 6 — (Recommended) Install GNU make

The Makefile gives you short commands like `make build` and `make smoke`. If
you can't install make, jump to Option B — every `make` target maps to a single
`docker compose` command you can run directly.

**Option A — winget (Windows 10 1709+ or Windows 11):**

```powershell
winget install --id GnuWin32.Make
```

You may need to restart your terminal afterward so it picks up the new
`PATH`. Verify with `make --version`.

If git bash still cannot find `make`, you may have to add manually the path
to its location. For instance, winget may have installed `make` in 
`C:\Program Files (x86)\GnuWin32\bin\`
In this case (check first if `make` is actually there),
create a local bash configuration script `~/.bashrc` in your 
user `$HOME` folder containing the following line:

```bash
   export PATH="$PATH:/c/Program Files (x86)/GnuWin32/bin"
```
You can do this for instance with a heredoc (or with your favourite text editor):

```bash
   cd $HOME
   cat > .bashrc << 'EOF'
   export PATH="$PATH:/c/Program Files (x86)/GnuWin32/bin"
   EOF
```

Restart the git bash terminal before trying again `make --version`.


**Option B — skip `make`.** Use this table:

| Instead of...      | Run this directly                                              |
| ------------------ | -------------------------------------------------------------- |
| `make build`       | `docker compose build`                                         |
| `make rebuild`     | `docker compose build --no-cache`                              |
| `make up`          | `docker compose up`                                            |
| `make down`        | `docker compose down`                                          |
| `make shell`       | `docker compose run --rm --service-ports abides bash`          |
| `make test`        | `docker compose run --rm abides pytest -q`                     |
| `make smoke`       | `docker compose run --rm abides python tools/smoke_test.py`    |
| `make lock`        | `docker compose run --rm abides bash -c "pip-compile --quiet --strip-extras --resolver=backtracking --output-file=requirements.lock requirements.in requirements-dev.in"` |
| `make clean`       | `docker compose down --rmi local --volumes --remove-orphans`   |

---

## Step 7 — Choose where to clone the repo

This matters more on Windows than other OSes. Avoid:

- Anything inside `OneDrive\` — OneDrive sync constantly touches the files
  and dramatically slows Docker volume I/O. You will see builds that take
  10× longer than necessary.
- Anything inside your `Documents` folder if Documents is itself
  OneDrive-synced (it often is by default).
- Paths with spaces or non-ASCII characters (`C:\Università\...` is asking
  for trouble — bash in the container may misquote them).
- Paths longer than 200 characters, especially with deeply nested folders.
  Windows has historical limits on `MAX_PATH` and Docker's bind-mount
  doesn't always handle them gracefully.

**Recommended location:** `C:\dev\abides-polimi`. Create the parent in Git
Bash:

```bash
mkdir -p /c/dev
cd /c/dev
```

(`/c/dev` in Git Bash is `C:\dev` in Explorer.)

---

## Step 8 — Clone the repo

In Git Bash, from `/c/dev` (or wherever you chose):

```bash
git clone <the-repo-url> abides-polimi
cd abides-polimi
```

Replace `<the-repo-url>` with the URL the teaching team gave you.

---

## Step 9 — Build the image and run the smoke test

This is the moment of truth. Make sure Docker Desktop is running (whale icon
in the system tray, not animated).

```bash
make build && make smoke
```

If you don't have `make` installed, run the same thing as two commands:

```bash
docker compose build
docker compose run --rm abides python tools/smoke_test.py
```

**What to expect:**

- The build will print a long stream of pip output as it installs JupyterLab,
  NumPy, pandas, scipy, gym, pomegranate, and the three ABIDES sub-packages.
  First run is 5–10 minutes on a typical laptop.
- The smoke test takes 10–30 seconds.
- On success you'll see something like:

  ```
  [smoke] Building RMSC04 config (5-minute simulation)...
  [smoke] Running simulation...
  [smoke] ABM: bids=… asks=… history=…
  [smoke] OK (12.3s) — environment looks healthy.
  ```

If the last line is `[smoke] OK …`, you're done with setup. Move on to the
**Daily workflow** section.

---

## Daily workflow

Once the image is built, day-to-day commands are fast (the image is cached,
your source is bind-mounted, and editable installs make code changes
immediately visible inside the container).

```bash
make up        # start JupyterLab on http://localhost:8888
make down      # stop the container
make shell     # bash shell inside the container
make test      # run the pytest suite
make smoke     # re-run the smoke test
```

Your edits in VS Code, PyCharm, or any editor on the host show up inside the
container immediately — no rebuild needed. Only changes to `Dockerfile`,
`requirements*.in`, or `requirements.lock` need a `make rebuild`.

If you prefer the **VS Code Dev Containers** flow:

1. Open the cloned folder in VS Code.
2. When prompted, click **Reopen in Container**, or use the command palette:
   *Dev Containers: Reopen in Container*.
3. VS Code builds and starts the container, mounts the source, and drops you
   into a terminal where everything just works.

---

## Troubleshooting (Windows-specific)

| Symptom                                                                | Cause / fix |
| ----------------------------------------------------------------------- | ----------- |
| `docker: command not found`                                             | Docker Desktop isn't running, or wasn't installed correctly. Open Docker Desktop from the Start menu and wait for the whale icon to settle. |
| `Cannot connect to the Docker daemon`                                   | Same as above — Docker engine isn't running. |
| Build hangs at `RUN apt-get update`                                     | Network / corporate firewall. Check that you can `curl https://deb.debian.org/debian/` from inside the container with `make shell`. |
| Build is glacially slow (>30 min)                                       | You probably cloned into OneDrive or Documents. Move the repo to `C:\dev\abides-polimi` and rebuild. |
| `pomegranate` build takes 5+ minutes during install                     | Expected on first build (pomegranate compiles from source on Windows/arm64). The result is cached for next time. |
| `port is already allocated` on 8888                                     | Another process is using port 8888. Stop it (`netstat -ano \| findstr :8888` to find the PID), or change the port in `docker-compose.yml`. |
| Edits in VS Code don't show up in the running container                 | File watching across the host/container boundary can lag on Windows. Save the file, then in the container run `touch <file>` to nudge it, or rebuild with `make rebuild`. |
| `make: command not found`                                               | See Step 6 — install via `winget`, or use the `docker compose` commands directly. |
| Antivirus blocks Docker (Defender, McAfee, etc. quarantining files)     | Add an exclusion for `C:\ProgramData\Docker\` and the repo folder. Corporate AV may need IT involvement. |
| `WSL2 installation is incomplete`                                       | Run `wsl --update` from an admin PowerShell. If it still fails, fully uninstall WSL with `wsl --unregister Ubuntu` and reinstall. |
| Hyper-V conflicts with VirtualBox / VMware                              | You can't run Hyper-V-based Docker Desktop and VirtualBox at the same time on older Windows. Either disable Hyper-V (`bcdedit /set hypervisorlaunchtype off` then reboot) for VirtualBox, or use the WSL2 backend (the default new behaviour). |
| `Mounts denied` errors on `student_work`                                | Make sure file sharing is enabled in Docker Desktop: Settings → Resources → File sharing → add your `C:\dev\` path. |

---

## Getting help

Before asking the teaching team for help, please include:

1. The exact command you ran.
2. The full error output (copy-paste, don't summarise).
3. The output of `docker --version` and `wsl --status`.
4. Where you cloned the repo (run `pwd` in Git Bash inside the repo folder).

This makes it 10× faster for the team to spot the issue.
