import cv2
import numpy as np
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
from hailo_platform.pyhailort.pyhailort import FormatOrder

def preprocess_image(image_path, input_shape):
    """
    Load image, resize to model input shape and convert BGR to RGB.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def parse_hef_output_xyxy(output_array, img_width, img_height):
    """
    Parse the HEF model output assuming XYXY normalized bounding boxes.
    output_array format per detection: 
    [x1_norm, y1_norm, x2_norm, y2_norm, confidence]
    no class_id sinse it's the only class
    """
    num_detections = int(output_array[0])
    detections = []
    offset = 1
    for i in range(num_detections):
        if offset + 5 >= len(output_array):
            break
        #Please note that instead of ONNX inference, HEF returns a relative sequence of Y1, X1, Y2, X2, confidence [and maybe class] in raw output
        #I don't know why the structure of output was changed this way
        y1 = output_array[offset]
        x1 = output_array[offset + 1]
        y2 = output_array[offset + 2]
        x2 = output_array[offset + 3]
        confidence = output_array[offset + 4]
        class_id = 0 #hardcoded to 0
        offset += 5

        x1_abs = int(x1 * img_width)
        y1_abs = int(y1 * img_height)
        x2_abs = int(x2 * img_width)
        y2_abs = int(y2 * img_height)

        detections.append({
            "bbox_xyxy": [x1_abs, y1_abs, x2_abs, y2_abs],
            "bbox_rel": [x1,y1,x2,y2],
            "confidence": confidence,
            "class_id": class_id
        })
    return detections

def draw_bboxes(image, detections, class_names=None):
    """
    Draw bounding boxes and labels on the image.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det["confidence"]
        cls = det["class_id"]
        label = f"{cls}"
        if class_names and cls < len(class_names):
            label = f"{cls} ({class_names[cls]})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def main():
    hef_path = "yolov8n.hef"
    image_path = "test.jpg"
    input_shape = (320, 320)
    class_names = ["fly"]  # Single class for this model

    # Load and preprocess image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    input_tensor = preprocess_image(image_path, input_shape)

    # Setup Hailo device and model
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    target = VDevice(params)
    infer_model = target.create_infer_model(hef_path)

    infer_model.set_batch_size(1)
    infer_model.input().set_format_type(FormatType.UINT8)

    output_buffers = {}
    for output_name in infer_model.output_names:
        shape = infer_model.output(output_name).shape
        output_buffers[output_name] = np.empty(shape, dtype=np.float32)
        infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

    # Run inference
    with infer_model.configure() as configured:
        bindings = configured.create_bindings(output_buffers=output_buffers)
        bindings.input().set_buffer(input_tensor)
        configured.run([bindings], timeout=5000)

    # Assuming model outputs in a single output layer with YOLO postprocess results
    output_name = infer_model.output_names[0]
    output_array = output_buffers[output_name].flatten()

    # Parse detections
    detections = parse_hef_output_xyxy(output_array, img_width, img_height)

    # Print detections info
    print(f"\nResults for HEF:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        rel_x1, rel_y1, rel_x2, rel_y2 = det["bbox_rel"]
        conf = det["confidence"]
        cls = det["class_id"]
        cls_name = class_names[cls] if cls < len(class_names) else "unknown"
        #print(f"Detection {i}: class={cls} ({cls_name}), confidence={conf:.2f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
        print(f"Box {i} coords(calculated): [{x1}, {y1}, {x2}, {y2}], confidence={conf:.2f}.")
        print(f"Box {i} relative(native output)=({rel_x1:.3f}, {rel_y1:.3f}, {rel_x2:.3f}, {rel_y2:.3f})")


    # Draw bounding boxes on original image (BGR for display)
    draw_bboxes(image, detections, class_names)

    # Show image with boxes
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
