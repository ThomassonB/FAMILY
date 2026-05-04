from pathlib import Path
import subprocess

def run_julia_export(
    julia_script: str,
    input_file: str,
    fits_out: str,
    csv_outdir: str,
    nxy: int = 400,
    julia_bin: str = "julia",
) -> None:
    cmd = [
        julia_bin,
        julia_script,
        input_file,
        fits_out,
        csv_outdir,
        str(nxy),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)