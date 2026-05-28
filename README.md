<div id="top"></div>

# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

> **Course note:** this is the Politecnico fork used for part 3, *Build and
> Battle Your Trading Agent*, of the algorithmic trading course. If you are a
> student, jump straight to **[Quickstart for Students](#quickstart-for-students)**
> below — it gives you a working environment in three commands on Mac or Windows.

<!-- TABLE OF CONTENTS -->
<ol>
  <li><a href="#quickstart-for-students">Quickstart for Students</a></li>
  <li>
    <a href="#about-the-project">About The Project</a>
  </li>
  <li><a href="#citing-abides">Citing ABIDES</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li>
    <a href="#usage-regular">Usage (regular)</a>
    <ul>
      <li><a href="#using-a-python-script">Using a Python Script</a></li>
      <li><a href="#using-the-abides-command">Using the `abides` Command</a></li>
    </ul>
  </li>
  <li><a href="#usage-gym">Usage (Gym)</a></li>
  <li><a href="#default-available-markets-configurations">Default Available Markets Configurations</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

<!-- QUICKSTART -->
## Quickstart for Students

The lab runs inside a Docker container, so the environment is identical on
Intel Macs, Apple Silicon Macs, and Windows. You install Docker once, then
everything else lives inside the container.

**For first-time setup, follow the step-by-step guide for your OS:**

- 🪟 **Windows** — [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)
- 🍎 **macOS** (Intel and Apple Silicon) — [docs/SETUP_MACOS.md](docs/SETUP_MACOS.md)

Once you've installed the prerequisites (Docker Desktop, Git, optionally
VS Code with the *Dev Containers* extension), the day-to-day flow is:

```bash
git clone <this-repo-url> abides-polimi
cd abides-polimi
make build && make smoke    # one-time build + verification
```

`make build` builds the Docker image (5–10 minutes the first time, cached
after). `make smoke` runs a 5-minute RMSC04 simulation and confirms the
environment is healthy — look for `[smoke] OK … environment looks healthy.`

### Day-to-day commands

| Task                                  | Command         |
| ------------------------------------- | --------------- |
| Start JupyterLab on `localhost:8888`  | `make up`       |
| Stop the container                    | `make down`     |
| Drop into a shell inside the container| `make shell`    |
| Run the unit tests                    | `make test`     |
| Run the smoke test                    | `make smoke`    |
| Full reset (rebuild image)            | `make clean && make build` |

The whole project source is bind-mounted into the container, so editing files
on your host (in VS Code, PyCharm, anything) is picked up immediately by the
running container — no rebuild needed.

### VS Code one-click flow

1. Open the cloned repo folder in VS Code.
2. When prompted, choose **Reopen in Container** (or run *Dev Containers:
   Reopen in Container* from the command palette).
3. VS Code builds the image, mounts the source, installs the recommended
   Python and Jupyter extensions inside the container, and drops you into a
   terminal where `python`, `pytest`, and `make smoke` all work.

For OS-specific troubleshooting, see the per-OS setup guide linked above.

### Dependency management

Each sub-package (`abides-core`, `abides-markets`, `abides-gym`) declares its
own runtime dependencies in its own `pyproject.toml` (PEP 621
`[project].dependencies` block). Top-level classroom extras (JupyterLab,
matplotlib) live in `requirements.in`. Dev tooling (pytest, mypy, sphinx)
lives in `requirements-dev.in`. Tool config (mypy, black, isort, pytest)
lives in the repo-level `pyproject.toml`.

For reproducibility, a fully-pinned `requirements.lock` is generated from
`requirements.in` + `requirements-dev.in`. The Docker build prefers the
lockfile when it exists; if not, it falls back to the looser `.in` files.

To regenerate the lockfile after changing any dependency:

```bash
make lock     # runs `uv pip compile` inside the container
git diff requirements.lock      # review
git add requirements.lock && git commit
```

Students who only consume the project never need to run `make lock` — they
get the committed lockfile via `git pull` and `make build` will install from
it.


<!-- ABOUT THE PROJECT -->
## About The Project

ABIDES (Agent Based Interactive Discrete Event Simulator) is a general purpose multi-agent discrete event simulator. Agents exclusively communicate through an advanced messaging system that supports latency models.

The project is currently broken down into 3 parts: ABIDES-Core, ABIDES-Markets and ABIDES-Gym.

* ABIDES-Core: Core general purpose simulator that be used as a base to build simulations of various systems.
* ABIDES-Markets: Extension of ABIDES-Core to financial markets. Contains implementation of an exchange mimicking NASDAQ, stylised trading agents and configurations.
* ABIDES-Gym: Extra layer to wrap the simulator into an OpenAI Gym environment for reinforcement learning use. 2 ready to use trading environments available. Possibility to build other financial markets environments easily.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITING -->
## Citing ABIDES

[ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets](https://arxiv.org/pdf/2110.14771.pdf) or use
the following BibTeX:

```
@misc{amrouni2021abidesgym,
      title={ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets}, 
      author={Selim Amrouni and Aymeric Moulin and Jared Vann and Svitlana Vyetrenko and Tucker Balch and Manuela Veloso},
      year={2021},
      eprint={2110.14771},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```

[ABIDES: Towards High-Fidelity Market Simulation for AI Research](https://arxiv.org/abs/1904.12066)
or by using the following BibTeX:

```
@misc{byrd2019abides,
      title={ABIDES: Towards High-Fidelity Market Simulation for AI Research}, 
      author={David Byrd and Maria Hybinette and Tucker Hybinette Balch},
      year={2019},
      eprint={1904.12066},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Installation

For students taking the course, follow the
**[Quickstart for Students](#quickstart-for-students)** above — it uses
Docker and works identically on Mac and Windows.

If you specifically need a host install (e.g. you are extending ABIDES outside
the course), the legacy path is:

```bash
git clone <this-repo-url>
cd abides-polimi
sh install.sh                    # editable: sh setup-dev.sh
```

The legacy path requires Python 3.9 and a working build toolchain (Xcode CLT
on macOS, MSVC build tools on Windows) for `numba`, `llvmlite`, and
`pomegranate`. The Docker path avoids all of that.


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage (regular)
Regular ABIDES simulations can be run either directly in python or through the command line

_For more examples, please refer to the [Documentation](https://example.com)_

### Using a Python Script:

```python
from abides_markets.configs import rmsc04
from abides_core import abides

config_state = rmsc04.build_config(seed = 0, end_time = '10:00:00')
end_state = abides.run(config_state)
```
<p align="right">(<a href="#top">back to top</a>)</p>

### Using the abides Command:

The config can be loaded and the simulation run using the `abides`
command in the terminal (from directory root):

```
$ abides abides-markets/abides_markets/configs/rmsc04.py --end_time "10:00:00"
```

The first argument is a path to a valid ABIDES configuration file.

Any further arguments are optional and can be used to overwrite any parameters
in the config file.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage (Gym)
ABIDES can also be run through a Gym interface using ABIDES-Gym environments.

```python
import gym
import abides_gym

env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
)

env.seed(0)
initial_state = env.reset()
for i in range(5):
    state, reward, done, info = env.step(0)
```

## Default Available Markets Configurations

ABIDES currently has the following available background Market Simulation Configuration:

* RMSC03: 1 Exchange Agent, 1 POV Market Maker Agent, 100 Value Agents, 25 Momentum Agents, 5000 Noise Agents
 
* RMSC04: 1 Exchange Agent, 2 Market Maker Agents, 102 Value Agents, 12 Momentum Agents, 1000  Noise Agents

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

TODO: add information about JPMC contribution agreement

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License
Distributed under the BSD 3-Clause "New" or "Revised" License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
ABIDES was originally developed by David Byrd and Tucker Balch: https://github.com/abides-sim/abides
ABIDES is currently developed and maintained by [Jared Vann](https://github.com/jaredvann) (aka @jaredvann), [Selim Amrouni](https://github.com/selimamrouni) (aka @selimamrouni), and [Aymeric Moulin](https://github.com/AymericCAMoulin) (@AymericCAMoulin).
**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.

<p align="right">(<a href="#top">back to top</a>)</p>
