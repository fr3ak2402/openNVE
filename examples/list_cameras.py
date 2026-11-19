"""Lists the connected cameras and prints a JSON configuration for each one."""

import opennve

cameras = opennve.list_cameras()
print(opennve.list_cameras_json())

for camera in cameras:
    if camera.available:
        print(f"\nConfiguration for {camera.model} {camera.serial}:")
        print(camera.default_config().to_json())
