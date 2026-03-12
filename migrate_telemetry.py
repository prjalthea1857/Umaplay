import os
import shutil
import re
from pathlib import Path

telemetry_dir = Path("c:/Users/virt-walnutcrakr/Documents/Projects/Umaplay/datasets/telemetry")

for path in telemetry_dir.iterdir():
    if path.is_dir():
        # Match directory name like "Air Groove_20260311_103952"
        match = re.match(r"^(.*)_(\d{4})(\d{2})(\d{2})_(\d{6})$", path.name)
        if match:
            trainee_name = match.group(1)
            date_str = f"{match.group(2)}-{match.group(3)}-{match.group(4)}"
            run_id = f"{match.group(2)}{match.group(3)}{match.group(4)}_{match.group(5)}"
            
            dest_dir = telemetry_dir / trainee_name / date_str
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Migrating {path.name} -> {trainee_name}/{date_str}")
            
            for file in path.iterdir():
                if file.is_file():
                    target_file = dest_dir / file.name
                    if target_file.exists():
                        print(f"Target {target_file} already exists, skipping")
                    else:
                        shutil.move(str(file), str(target_file))
            
            try:
                path.rmdir()
            except OSError as e:
                print(f"Could not remove {path}: {e}")
