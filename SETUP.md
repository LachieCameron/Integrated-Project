# Setup (Mac and Windows)

Python 3.10 to 3.12 is recommended. Everyone should use a virtual environment so the team shares the same package versions.

## Get the code

```
git clone https://github.com/lachiecameron/tb3-ekf-lqr-nav.git
cd tb3-ekf-lqr-nav
```

## macOS (Terminal)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: the MuJoCo interactive viewer on macOS has to be launched with `mjpython` instead of `python`.

## Windows (PowerShell)

```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Team git workflow

- Run `git pull` before you start work each day.
- Do not commit straight to `main`. Make a branch per task, e.g. `part-c-ekf`.
- Open a pull request, have one teammate review it, then merge.
- Keep large files (videos, datasets) out of git and link them from the README.
- Write short, specific commit messages, e.g. `Add NIS gating to EKF update`.
- If you have not already, set your identity once: `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.
