import json
import os 
from collections import defaultdict


workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(workspace, "data")
if os.path.exists(f"{data_dir}/partnet_mobility_dict.json"):
    with open(f"{data_dir}/partnet_mobility_dict.json", 'r') as f:
        partnet_mobility_dict = json.load(f)
else:
    partnet_mobility_dict = {}

if os.path.exists("objaverse_utils/text_to_uid.json"):
    with open("objaverse_utils/text_to_uid.json", 'r') as f:
        text_to_uid_dict = json.load(f)
else:
    text_to_uid_dict = {}

if os.path.exists(f"{data_dir}/sapien_cannot_vhacd_part.json"):
    with open(f"{data_dir}/sapien_cannot_vhacd_part.json", 'r') as f:
        sapaien_cannot_vhacd_part_dict = json.load(f)
else:
    sapaien_cannot_vhacd_part_dict = defaultdict(list)


if __name__ == '__main__':
    print(partnet_mobility_dict["Oven"])