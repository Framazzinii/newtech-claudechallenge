# Setting up the ABIDES lab on macOS

This is the step-by-step setup guide for macOS, both Intel and Apple Silicon
(M-series) Macs. It assumes you have never installed Docker before. Allow
about 20–30 minutes for the first run, mostly waiting on downloads.

If you are on Windows, see [SETUP_WINDOWS.md](SETUP_WINDOWS.md) instead.

## What you'll install

1. Docker Desktop — runs the lab container.
2. Xcode Command Line Tools — gives you Git and the basic compiler toolchain
   (already installed on most developer Macs).
3. (Optional) GNU make — already installed via the Xcode Command Line Tools.
4. (Optional) Visual Studio Code with the *Dev Containers* extension —
   one-click reopen-in-container experience.

You do **not** need to install Python, NumPy, JupyterLab, or any of ABIDES's
dependencies on your host machine. Everything lives inside the container.

> **Note on Apple Silicon (M1/M2/M3/M4):** the lab image builds natively for
> arm64. You do not need Rosetta. The first build is slightly slower than on
> Intel because `pomegranate` does not ship a prebuilt arm64 wheel and has to
> compile from source, but the result is cached and subsequent builds are
> fast.

---

## Step 1 — Install Xcode Command Line Tools

Open Terminal (<kbd>⌘</kbd>+<kbd>Space</kbd>, type "Terminal") and run:

```bash
xcode-select --install
```

If a popup appears, click **Install** and accept the licence. If you instead
see `xcode-select: error: command line tools are already installed`, you're
done with this step.

This gives you `git`, `make`, `cc`, and the rest of the basics.

**Verify:**

```bash
git --version
make --version
```

---

## Step 2 — Install Docker Desktop

1. Download Docker Desktop for Mac from
   <https://www.docker.com/products/docker-desktop/>. **Choose the right
   build for your chip:**
   - **Apple Silicon** (M1/M2/M3/M4): pick the *Apple Silicon* download.
   - **Intel Mac**: pick the *Intel chip* download.
   To check which Mac you have:  → *About This Mac* → look at "Chip" or
   "Processor".
2. Open the downloaded `.dmg`, drag *Docker* into *Applications*, and run it
   from Launchpad.
3. Accept the licence agreement (free for personal/educational use).
4. Docker Desktop will ask for your password to install its helper. Enter it.
5. Wait until the whale icon in the menu bar is steady (not animated). The
   first launch can take a minute or two.

**Verify:**

```bash
docker --version
docker run --rm hello-world
```

The second command pulls a tiny test image and prints "Hello from Docker!".
If both work, Docker is healthy.

---

## Step 3 — Choose where to clone the repo

Anywhere works on macOS, but a few mild preferences:

- Avoid paths inside iCloud Drive (`~/Documents` and `~/Desktop` are
  iCloud-synced by default on many setups). iCloud sync slows file watching
  the same way OneDrive does on Windows, just less dramatically.
- Avoid paths with spaces or non-ASCII characters
  (`~/Università/...` works but is unnecessary friction).

**Recommended location:** `~/dev/abides-polimi`.

```bash
mkdir -p ~/dev
cd ~/dev
```

---

## Step 4 — Clone the repo

```bash
git clone <the-repo-url> abides-polimi
cd abides-polimi
```

Replace `<the-repo-url>` with the URL the teaching team gave you.

---

## Step 5 — Build the image and run the smoke test

This is the moment of truth. Make sure Docker Desktop is running (whale icon
in the menu bar, not animated).

```bash
make build && make smoke
```

If you don't want to use `make` for some reason:

```bash
docker compose build
docker compose run --rm abides python tools/smoke_test.py
```

**What to expect:**

- The build prints a long stream of pip output as it installs JupyterLab,
  NumPy, pandas, scipy, gym, pomegranate, and the three ABIDES sub-packages.
  First run is 5–10 minutes (Intel) or 8–12 minutes (Apple Silicon, mostly
  spent compiling pomegranate from source).
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

## Troubleshooting (macOS-specific)

| Symptom                                                                  | Cause / fix |
| ------------------------------------------------------------------------- | ----------- |
| `docker: command not found`                                               | Docker Desktop isn't running. Open it from Applications and wait for the whale icon to settle. |
| `Cannot connect to the Docker daemon`                                     | Same as above — Docker engine isn't running. |
| `pomegranate` build takes 5–10 minutes on Apple Silicon                   | Expected on first build (no arm64 wheel published; compiles from source). Cached for next time. |
| `port is already allocated` on 8888                                       | Another process is using 8888. Find it with `lsof -i :8888` and stop it, or change the port in `docker-compose.yml`. |
| Build is unusually slow                                                   | Check that the repo is not in iCloud Drive. Move it to `~/dev/abides-polimi`. |
| Edits in your editor don't show up in the running container               | Save the file, then in the container run `touch <file>` to nudge file watching, or `make rebuild`. |
| Docker Desktop says "running but not responding"                          | Restart Docker Desktop:  → *Restart…*. If that fails, *Reset to factory defaults* (you'll lose container state but not source). |
| `Filesystem performance` warning in Docker Desktop                        | Docker Desktop's gRPC FUSE / VirtioFS settings affect bind-mount speed. Try Settings → General → *Use VirtioFS for file sharing* (faster on macOS 12.5+). |
| Old Macs without AVX (e.g. pre-2013): build fails on TensorFlow-adjacent  | Not relevant for the lab as configured (no TF dependency). If you've added one yourself, you'll need a different image base. |

---

## Getting help

Before asking the teaching team for help, please include:

1. The exact command you ran.
2. The full error output (copy-paste, don't summarise).
3. The output of `docker --version` and `sw_vers` (the macOS version).
4. Whether you're on Intel or Apple Silicon (`uname -m` — `x86_64` is Intel,
   `arm64` is Apple Silicon).
5. Where you cloned the repo (run `pwd` inside the repo folder).

This makes it 10× faster for the team to spot the issue.
