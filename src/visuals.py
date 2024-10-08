import cv2
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

OFFSET = 10
CIRCLE_RADIUS = 5
TIP_SIZE = 25


def visualize_graph(image_pil, nodes, edges, save_path=None):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes and node indices
    for idx, node in enumerate(nodes):
        box = node["bbox"]
        # Draw rectangle
        cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        # Calculate the position for the text (upper right corner)
        text_position = (box[2], box[1] + 20)  # Adjust the offset as needed
        # Put text at the calculated position
        cv2.putText(
            image_cv,
            str(idx),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw edges without duplication
    drawn_edges = set()
    for edge in edges:
        if (edge[1], edge[0]) in drawn_edges:
            continue  # Skip if the reverse edge is already drawn

        box_i = nodes[edge[0]]["bbox"]
        box_j = nodes[edge[1]]["bbox"]
        center_i = ((box_i[0] + box_i[2]) // 2, (box_i[1] + box_i[3]) // 2)
        center_j = ((box_j[0] + box_j[2]) // 2, (box_j[1] + box_j[3]) // 2)

        # Add small random offset to avoid overlapping lines
        offset_i = (random.randint(-OFFSET, OFFSET), random.randint(-OFFSET, OFFSET))
        offset_j = (random.randint(-OFFSET, OFFSET), random.randint(-OFFSET, OFFSET))
        start = (center_i[0] + offset_i[0], center_i[1] + offset_i[1])
        end = (center_j[0] + offset_j[0], center_j[1] + offset_j[1])

        # Calculate the line length
        line_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)                        
        # Calculate tipLength as a ratio of TIP_SIZE to line_length
        tip_length = min(TIP_SIZE / line_length, 0.3)  # Cap at 0.3 to avoid overly large arrow tips
        
        cv2.arrowedLine(image_cv, start, end, color=(0, 255, 0), thickness=2, tipLength=tip_length)

        # Draw dots at the ends of the edges
        cv2.circle(image_cv, start, radius=CIRCLE_RADIUS, color=(255, 0, 0), thickness=-1)
        cv2.circle(image_cv, end, radius=CIRCLE_RADIUS, color=(255, 0, 0), thickness=-1)

        # Mark this edge as drawn
        drawn_edges.add((edge[0], edge[1]))

    # Convert OpenCV image back to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    image_pil_width, image_pil_height = image_pil.size
    ratio = image_pil_height / image_pil_width

    # Display the image using matplotlib
    fig, ax = plt.subplots(figsize=(12, 12 * ratio))
    ax.imshow(image_pil)
    ax.axis('off')  # Turn off axis labels and ticks
    
    # Remove white space and adjust layout
    fig.tight_layout(pad=0)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()