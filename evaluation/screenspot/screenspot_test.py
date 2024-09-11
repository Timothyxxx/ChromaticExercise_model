import argparse
import json
import logging
import os

# Qwen2-VL-Chat settings
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from process_utils import pred_2_point, extract_bbox

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
parser.add_argument('--lora_path', type=str, default='Qwen2-VL-Chat')
parser.add_argument('--screenspot_imgs', type=str, default='screenspot_imgs')
parser.add_argument('--screenspot_test', type=str, default='screenspot_test')
parser.add_argument('--task', type=str, default='all')
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     args.qwen_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
#     trust_remote_code=True,
# ).eval()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.qwen_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
).eval()

processor = AutoProcessor.from_pretrained(args.qwen_path)

print("Load Success")

if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = []
result = []
batch_size = args.batch_size
for task in tasks:
    dataset = "screenspot_" + task + ".json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))
    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    for j, item in enumerate(tqdm(screenspot_data)):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

        system_prompt = "You are presented with a screenshot where the x-axis represents the width of the screen (increasing from left to right) and the y-axis represents the height (increasing from top to bottom). The origin (0, 0) is located at the top-left corner of the image. You response should formatted as `<box>(x1, y1), (x2, y2)</box>` (or w.o. <box> tag) where (x1, y1) is coodinates of the left top point of the element and (x2, y2) is the bottom right one. Or the mid-point of the element which formatted as `(x, y)`."
        question_prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {
                        "type": "text",
                        "text": question_prompt.format(instruction)
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        try:
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
            else:
                click_point = pred_2_point(response)

            print("click_point: ", click_point)
            print("bbox: ", bbox)

            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                logging.info("match " + str(corr_action / num_action))
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("unmatch " + str(corr_action / num_action))
            result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                           "type": item["data_type"], "source": item["data_source"]})
        except:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("Step: " + str(j) + " wrong format")

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)
