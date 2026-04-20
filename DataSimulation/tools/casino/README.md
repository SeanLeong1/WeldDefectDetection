# CASINO Adapter + AutoHotkey Runner

This folder contains the adapter plus an AutoHotkey-driven external runner for CASINO GUI automation.

## Files

- `casino_adapter.py` : adapter implementation
- `casino_adapter.cmd` : Windows launcher (recommended for `--casino-exe`)
- `casino_runner_template.py` : AutoHotkey-driven external runner
- `casino_ahk_profile.example.json` : editable action/output profile template

## Adapter Contract

The validation pipeline calls:

```text
casino_adapter --version
casino_adapter --input <case.json> --output <casino_output.json>
```

The output file must contain:

```json
{
  "quadrants": {
    "A": 0.1,
    "B": 0.1,
    "C": 0.1,
    "D": 0.1
  }
}
```

## Current Modes

- `mock` (default): writes deterministic synthetic quadrants for wiring tests only.
- `external`: runs CASINO via runner and parses quadrant outputs to write `casino_output.json`.

Set mode with:

```powershell
$env:BSE_CASINO_ADAPTER_MODE="mock"
```

or:

```powershell
$env:BSE_CASINO_ADAPTER_MODE="external"
```

## External Mode Options

- `BSE_CASINO_WIN_EXE` : path to `wincasino2_64.exe` (or pass `--casino-exe`).
- `BSE_CASINO_EXTERNAL_RUNNER_SCRIPT` : path to Python runner script (recommended).
- `BSE_CASINO_EXTERNAL_RUNNER_ARGS` : extra args appended to runner script call.
- `BSE_AUTOHOTKEY_EXE` : path to `AutoHotkey64.exe` (or pass `--ahk-exe` through runner args).
- `BSE_CASINO_EXTERNAL_RUNNER` : optional command template. Supports placeholders:
  - `{input}` case JSON path
  - `{output}` output JSON path
  - `{work_dir}` working directory for the case
  - `{casino_exe}` CASINO executable path
- `BSE_CASINO_TIMEOUT_S` : timeout in seconds (default `600`).

If a runner command is not provided, adapter directly executes CASINO executable with optional `BSE_CASINO_WIN_ARGS`.

## Quick Wiring Test

```powershell
$env:BSE_CASINO_EXE="C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_adapter.cmd"
$env:BSE_CASINO_ADAPTER_MODE="mock"
python main.py validation --casino-calibrate --casino-budget fast
```

Then confirm in the latest `validation_report.json`:

- `casino_probe.available = true`
- `calibration.status = ok` or `partial`

## Real CASINO + AHK Setup

1. Install AutoHotkey (v2 recommended), then confirm executable path (for example):
   - `C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe`
2. Copy and customize profile:

```powershell
Copy-Item `
  "C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_ahk_profile.example.json" `
  "C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_ahk_profile.json"
```

3. Edit `casino_ahk_profile.json`:
   - Set `"template_profile": false`
   - Update click coordinates and key steps in `actions` to match your CASINO GUI layout
   - Set `output.path` and parser settings to where your automation exports `A,B,C,D`

4. Configure env vars:

```powershell
$env:BSE_CASINO_ADAPTER_MODE="external"
$env:BSE_CASINO_WIN_EXE="C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\CASINO_v2.5.1.0\wincasino2_64.exe"
$env:BSE_CASINO_EXTERNAL_RUNNER_SCRIPT="C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_runner_template.py"
$env:BSE_CASINO_EXTERNAL_RUNNER_ARGS="--profile C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_ahk_profile.json --timeout-s 900 --keep-script"
$env:BSE_AUTOHOTKEY_EXE="C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"
```

5. Run validation:

```powershell
$env:BSE_CASINO_EXE="C:\Users\Sean Leong\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\tools\casino\casino_adapter.cmd"
python main.py validation --casino-calibrate --casino-budget fast
```

## Output Parser Formats

Set `output.format` in profile to one of:

- `csv_columns`:
  - file contains columns for `A,B,C,D` (mapped via `output.columns`)
- `json`:
  - file contains keys mapped via `output.keys` (default `quadrants.A` etc.)
- `regex`:
  - parse from text file using regex captures per quadrant
- `last4numbers`:
  - fallback parser: last 4 numeric values in file are treated as `A,B,C,D`

The runner writes normalized adapter output to `{output}` for the validation pipeline.

## First Bring-Up Tips

- Start with a single-case call first:
  - `python tools/casino/casino_runner_template.py --case <case.json> --out <casino_output.json> --work <dir> --casino <wincasino2_64.exe> --profile <casino_ahk_profile.json> --keep-script`
- Keep `--keep-script` enabled while tuning so you can inspect generated `.ahk`.
- Use `--emit-mock` only to test plumbing, not physics.
