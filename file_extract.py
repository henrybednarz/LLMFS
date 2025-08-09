import subprocess
import shlex


test_file = "./testdir/OfferLetter.pdf"


def mac_metadata(path):
    cmd = f"mdls {shlex.quote(path)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"mdls failed: {result.stderr.strip()}")

    raw_output = result.stdout
    x = raw_output.split("kMD")

    metadata = {}

    return metadata


print(mac_metadata(test_file))