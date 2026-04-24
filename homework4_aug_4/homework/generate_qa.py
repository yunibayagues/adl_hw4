import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    center_x_coord = img_width /2
    center_y_coord = img_height/2
    img_center = (int(img_width / 2), int(img_height / 2))

    with open(info_path, 'r') as file:
        info = json.load(file)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        return []

    kart_names = info['karts']

    karts = []
    distances = []

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if class_id == 1:  # it's a kart
            kart_dict_temp = {}
            if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled-y1_scaled) < min_box_size:
                continue # kart too small, out of sight out of mind
            if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
                continue

            if track_id < len(kart_names):
                name = kart_names[track_id]
            else:
                name = f"kart_{track_id}"

            kart_dict_temp['instance_id'] = track_id
            kart_dict_temp["kart_name"] = name
            x_coord = (x1_scaled + x2_scaled)/2
            y_coord = (y1_scaled + y2_scaled)/2
            center = (x_coord, y_coord)
            kart_dict_temp["center"] = center
            kart_dict_temp["is_center_kart"] = False
            distance_from_center = np.sqrt((center[0]-img_center[0])**2 + (center[1]-img_center[1])**2)
            distances.append(distance_from_center)
            karts.append(kart_dict_temp)
    if len(karts) == 0:
        return []
    distances = np.array(distances)
    min_index = np.argmin(distances)

    karts[min_index]['is_center_kart'] = True

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path, 'r') as file:
        info = json.load(file)

    return info['track']


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs_list = []
    p = Path(info_path)
    prefix = p.stem.split("_")[0]
    img_path = f"{Path(info_path).parent.name}/{prefix}_{view_index:02d}_im.jpg"

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    track = extract_track_info(info_path)

    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego is None:
        return []

    ego_x, ego_y = ego["center"]

    # 1. Ego car question
    # What kart is the ego car?

    ego_name = ego["kart_name"]

    qa_pairs_list.append({
        "question": "What kart is the ego car?",
        "answer": ego_name,
        "image_file": img_path
    })

    # 2. Total karts question
    # How many karts are there in the scenario?

    kart_count = len(karts)

    qa_pairs_list.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(kart_count),
        "image_file": img_path
    })

    # 3. Track information questions
    # What track is this?

    qa_pairs_list.append({
        "question": "What track is this?",
        "answer": track,
        "image_file": img_path
    })

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    left = 0
    right = 0
    front = 0
    behind = 0

    for kart in karts:
        if kart['is_center_kart']:
            continue
        name = kart['kart_name']
        kart_x, kart_y = kart['center']
        dx = kart_x - ego_x
        dy = kart_y - ego_y

        lr = "left" if dx < 0 else "right"
        fb = 'front' if dy < 0 else "back"

        if dx < 0:
            left += 1
        else:
            right += 1

        if dy < 0:
            front += 1
        else:
            behind += 1

        qa_pairs_list.append({
            "question": f"Is {name} to the left or right of the ego car?",
            "answer": lr,
            "image_file": img_path
        })

        qa_pairs_list.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": fb,
            "image_file": img_path
        })

        qa_pairs_list.append({
            "question": f"Where is {name} relative to the ego car?",
            "answer": f"{fb} and {lr}",
            "image_file": img_path
        })

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs_list.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left),
        "image_file": img_path
    })

    qa_pairs_list.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right),
        "image_file": img_path
    })

    qa_pairs_list.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front),
        "image_file": img_path
    })

    qa_pairs_list.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind),
        "image_file": img_path
    })

    return qa_pairs_list

def generate_all(data_dir: str = "data/train", output_file: str = "qa_pairs.json"):
    """
    Generate QA pairs for ALL images/views in dataset.
    """
    data_dir = Path(data_dir)
    output_file = data_dir / output_file

    all_qa_pairs = []

    data_dir = Path(data_dir)

    info_files = sorted(data_dir.rglob("*_info.json"))

    for info_file in info_files:
        with open(info_file, "r") as f:
            info = json.load(f)

        num_views = len(info["detections"])

        for view_index in range(num_views):
            qa_pairs = generate_qa_pairs(
                str(info_file),
                view_index=view_index
            )

            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)

    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f)

    print(f"Generated {len(all_qa_pairs)} QA pairs → {output_file}")


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def load_valid_grader(path="data/valid_grader/balanced_qa_pairs.json"):
    with open(path, "r") as f:
        return json.load(f)

ROOT = Path(__file__).resolve().parent.parent

def image_to_info_path(image_file):
    p = Path(image_file)
    base = p.stem.split("_")[0]

    return ROOT / "data" / "valid" / f"{base}_info.json"

def extract_view_index(image_file: str) -> int:
    return int(Path(image_file).stem.split("_")[1])

def check_against_valid_grader(example, img_width=150, img_height=100):
    """
    Compare your generated QA output vs ground truth for one sample.
    """

    image_file = example["image_file"]
    question = example["question"]
    gt_answer = example["answer"]

    info_path = image_to_info_path(image_file)

    # run your pipeline pieces
    karts = extract_kart_objects(str(info_path), view_index=extract_view_index(image_file),
                                 img_width=img_width, img_height=img_height)

    track = extract_track_info(str(info_path))

    qa_pairs = generate_qa_pairs(str(info_path),
                                 view_index=extract_view_index(image_file),
                                 img_width=img_width,
                                 img_height=img_height)

    # find YOUR answer for same question
    pred_answer = None
    for qa in qa_pairs:
        if qa["question"] == question:
            pred_answer = qa["answer"]
            break

    return {
        "question": question,
        "gt": gt_answer,
        "pred": pred_answer,
        "match": pred_answer == gt_answer
    }

def evaluate_against_grader(n=50):
    data = load_valid_grader()

    results = []

    for i, ex in enumerate(data[:n]):
        res = check_against_valid_grader(ex)
        results.append(res)

        print(res)

    acc = sum(r["match"] for r in results) / len(results)
    print(f"\nAccuracy: {acc:.4f}")

    return results

def evaluate_against_valid_grader(n=None):
    """
    Evaluate QA pipeline against official valid_grader dataset.
    """

    data = load_valid_grader()

    if n is not None:
        data = data[:n]

    total = 0
    correct = 0

    for ex in data:
        image_file = ex["image_file"]
        question = ex["question"]
        gt_answer = ex["answer"]

        info_path = image_to_info_path(image_file)
        view_index = extract_view_index(image_file)

        qa_pairs = generate_qa_pairs(
            str(info_path),
            view_index=view_index
        )

        # build lookup table
        pred_map = {qa["question"]: qa["answer"] for qa in qa_pairs}

        pred_answer = pred_map.get(question)

        match = (pred_answer == gt_answer)

        total += 1
        correct += int(match)

        if not match:
            print("Mismatch:")
            print("Q:", question)
            print("GT:", gt_answer)
            print("Pred:", pred_answer)
            print("-" * 40)

    accuracy = correct / total if total > 0 else 0.0

    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate_all": generate_all,
        "eval_valid": evaluate_against_valid_grader,
        "check_one": check_against_valid_grader
    })


if __name__ == "__main__":
    main()
