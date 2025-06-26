import subprocess

import modal

app = modal.App(name="timecopilot.dev")


@app.function(
    image=modal.Image.debian_slim()
    .add_local_dir("site", remote_path="/root/site", copy=True)
    .workdir("/root/site")
)
@modal.web_server(8000, custom_domains=["timecopilot.dev"])
def run():
    cmd = "python -m http.server 8000"
    subprocess.Popen(cmd, shell=True)
