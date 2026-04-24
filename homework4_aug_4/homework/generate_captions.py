from pathlib import Path

import fire
import json
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_track_info, extract_kart_objects

def image_to_info_path(image_file: str):
    p = Path(image_file)
    base = p.stem.split("_")[0]  # e.g. 000d9

    return Path("data") / "valid" / f"{base}_info.json"


def extract_view_index(image_file: str):
    return int(Path(image_file).stem.split("_")[1])

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if len(karts) == 0:
        return []

    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego is None:
        return []

    ego_name = ego["kart_name"]
    num_karts = len(karts)

    ego_x, ego_y = ego["center"]

    left = right = front = behind = 0
    captions = []

    # 1. Ego caption
    captions.append(f"{ego_name} is the ego car.")

    # 2. Count caption
    captions.append(f"There are {num_karts} karts in the scene.")

    # 3. Track caption
    captions.append(f"The track is {track}.")

    # 4. Relative captions + counting
    for kart in karts:
        if kart["is_center_kart"]:
            continue

        name = kart["kart_name"]
        x, y = kart["center"]

        dx = x - ego_x
        dy = y - ego_y

        lr = "left" if dx < 0 else "right"
        fb = 'in front of' if dy < 0 else "behind"

        captions.append(f"{name} is {lr} of the ego car.")
        captions.append(f"{name} is {fb} the ego car.")

        if dx <= 0:
            left += 1
        else:
            right += 1

        if dy < 0:
            front += 1
        else:
            behind += 1

    captions.append(f"There are {left} karts to the left of the ego car.")
    captions.append(f"There are {right} karts to the right of the ego car.")
    captions.append(f"There are {front} karts in front of the ego car.")
    captions.append(f"There are {behind} karts behind the ego car.")

    return captions

    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

from pathlib import Path
import json

def generate_all_train(data_dir: str = "data/train",
                       output_file: str = "data/train/all_captions.json"):

    data_dir = Path(data_dir)

    info_files = sorted(data_dir.rglob("*_info.json"))

    all_outputs = []

    for info_file in info_files:
        # load metadata
        with open(info_file, "r") as f:
            info = json.load(f)

        num_views = len(info["detections"])

        for view_index in range(num_views):
            captions = generate_caption(str(info_file), view_index)

            all_outputs.append({
                "info_file": str(info_file),
                "view_index": view_index,
                "captions": captions
            })

    # write output
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"Generated {len(all_outputs)} caption sets → {output_file}")


def load_valid():
    with open("data/valid_grader/all_mc_qas.json", "r") as f:
        return json.load(f)

def generate_for_image(image_file):
    info_path = image_to_info_path(image_file)
    view_index = extract_view_index(image_file)

    return generate_caption(str(info_path), view_index)

def evaluate():
    data = load_valid()

    total = 0
    correct = 0

    for ex in data:
        image_file = ex["image_file"]
        gt_caption = ex["candidates"][ex["correct_index"]]

        preds = generate_for_image(image_file)

        match = gt_caption in preds

        if match:
            correct += 1
        else:
            print("\n----------------------------------------")
            print("MISMATCH")
            print(f"Image: {image_file}")
            print(f"GT: {gt_caption}")
            print("Preds:")
            for p in preds:
                print(f"  - {p}")

        total += 1

    acc = correct / total
    print(f"\nAccuracy: {acc:.4f} ({correct}/{total})")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption,
               "evaluate": evaluate,
               "generate_all": generate_all_train})


if __name__ == "__main__":
    main()
