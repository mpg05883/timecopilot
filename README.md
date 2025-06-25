<div align="center">
  <a href="[https://ai.pydantic.dev/](https://github.com/AzulGarza/TimeCopilot)">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/27fdd7c8-483e-4339-bc54-23323582b39d">
      <img src="https://github.com/user-attachments/assets/7fdba4f2-e279-4fdf-b559-2829b5fe2143" alt="TimeCopilot">
    </picture>
  </a>
</div>
<div align="center">
  <em>The GenAI Forecasting Agent · LLMs × Foundation Time Series Models</em>
</div>
<div align="center">
  <a href="https://github.com/AzulGarza/TimeCopilot/actions/workflows/ci.yaml"><img src="https://github.com/AzulGarza/TimeCopilot/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/timecopilot"><img src="https://img.shields.io/pypi/v/timecopilot.svg" alt="PyPI"></a>
  <a href="https://github.com/AzulGarza/timecopilot"><img src="https://img.shields.io/pypi/pyversions/timecopilot.svg" alt="versions"></a>
  <a href="https://github.com/AzulGarza/timecopilot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/azulgarza/timecopilot.svg?v" alt="license"></a>
  <a href="https://discord.gg/7GEdHR6Pfg"><img src="https://img.shields.io/discord/1387291858513821776?label=discord" alt="Join Discord" /></a>
</div>

---


# python-project-template  
> a simple template for organizing any python project.

## setup & environment

this template is designed to streamline the setup of any python project. follow these steps to get started:

### 1. install [`uv`](https://github.com/astral-sh/uv/)

use `uv` to manage your virtual environments:

```bash
pip install uv
```

### 2. create a virtual environment

create an environment (or your preferred version):

```bash
uv venv
```

### 3. activate the environment

activate the virtual environment:

```bash
source .venv/bin/activate
```

### 4. install dependencies

install all the required dependencies from the `requirements.txt` file:

```bash
uv pip install -e .
```

### 5. install pre-commits

```bash
pre-commit install
```

## ready to code  
you're now all set to start building your project.
