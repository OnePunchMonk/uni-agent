# GUI agent example

A minimal desktop for the `gui_*` tools (`screenshot`, `click`, `type_text`, `press_key`,
`scroll`) and the `gui` task. The image runs Xvfb with `xdotool` and ImageMagick; the
test app is a Tk window whose button writes a marker file.

```bash
docker build -t uni-agent-gui examples/gui
```

Task sample (`gui` task, ReAct agent, docker sandbox with the image above):

```python
{
    "name": "gui",
    "sandbox": {"provider": "docker", "image": "uni-agent-gui"},
    "agent": {"name": "react", "tools": [{"name": n} for n in
              ("gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll")]},
    "setup_command": "nohup python3 /opt/gui/button_app.py >/tmp/app.log 2>&1 &",
    "verify_command": "test -f /tmp/gui_button_clicked",
    "metadata": {"instruction": "Click the SUBMIT button."},
}
```

The agent needs a vision-language model behind its endpoint. Screenshots reach it as a
user message with `image_url` parts after each turn's tool results.
