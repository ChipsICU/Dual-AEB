from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from random import shuffle
import torch, os
import json
from PIL import Image
import copy
import warnings
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from prettytable import PrettyTable
import argparse
import gc
from tqdm import tqdm
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on dataset chunks with multiple GPUs.")
    parser.add_argument('--gpu_id', type=int, required=True, help='ID of the GPU to use.')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--test_dataset_chunk', type=str, required=True, help='Path to the dataset chunk JSON file.')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for saving results.')
    parser.add_argument('--is_b2d', type=bool, default=True, help='Flag indicating whether to use B2D mode.')
    parser.add_argument('--num', type=int, required=True, help='Number of the GPU.')
    return parser.parse_args()

def load_model(pretrained, model_name, device_map="auto", device="cuda", overwrite_config=None):
    if overwrite_config is None:
        overwrite_config = {'tie_word_embeddings': True, 'use_cache': True}

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, model_name, device_map=device_map, overwrite_config=overwrite_config
    )
    model.to(device)
    model.eval()
    return tokenizer, model, image_processor

def extract_frames(video_path):
    frames = []
    for frame_path in video_path:
        try:
            with Image.open(frame_path) as img:
                frames.append(img.convert("RGB"))
        except IOError:
            print(f"Failed to read frame at path: {frame_path}")
    return frames

def generate_text(model, tokenizer, prompt_question, image_tensors, image_sizes, output_actions=False, device="cuda"):
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_actions=output_actions
    )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

def calculate_bleu4(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

def calculate_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def calculate_meteor(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return meteor_score([reference_tokens], candidate_tokens)

def save_conversations(conversations, output_path):
    with open(output_path, "w") as outfile:
        json.dump(conversations, outfile, indent=4)

def save_results_as_table(results, filename="evaluation_results.csv"):
    table = PrettyTable()
    table.field_names = ["Metric", "Average Score"]
    
    metrics = [
        "description_bleu4", "description_rouge_l", "description_meteor", 
        "bbox_bleu4", "bbox_rouge_l", "bbox_meteor",
        "action_bleu4", "action_rouge_l", "action_meteor",
    ]
    
    metric_sums = {metric: 0 for metric in metrics}
    num_results = len(results) - 1  # Exclude the overall metrics result
    
    # Sum up each metric for all results except the overall metrics
    for result in results:
        if result["id"] != "overall":  # Check to ensure it doesn't process the overall metrics
            for metric in metrics:
                metric_sums[metric] += result[metric]
    
    # Calculate the average for each metric and add it to the table
    for metric in metrics:
        average_score = metric_sums[metric] / num_results
        table.add_row([metric, f"{average_score:.4f}"])
    
    # Now add the overall precision and recall manually
    overall_result = [result for result in results if result["id"] == "overall"][0]
    table.add_row(["action_precision", f"{overall_result['action_precision']:.4f}"])
    table.add_row(["action_recall", f"{overall_result['action_recall']:.4f}"])
    
    # Write the table to a CSV file
    with open(filename, "w") as file:
        file.write(table.get_csv_string())

def calculate_precision_recall(action_answer_gt, action_answer_pred):
    gt_positive = "brak" in action_answer_gt.lower()
    pred_positive = "brak" in action_answer_pred.lower()

    true_positive = gt_positive and pred_positive
    false_positive = not gt_positive and pred_positive
    false_negative = gt_positive and not pred_positive

    return true_positive, false_positive, false_negative


def main(is_b2d=True):
    model_name = "llava_qwen"
    tokenizer, model, image_processor = load_model(args.pretrained, model_name)

    with open(args.test_dataset_chunk, "r") as f:
        test_dataset_chunk = json.load(f)
    results = []
    conversations = []
    
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    for sample in tqdm(test_dataset_chunk, desc=f"Evaluating on GPU {args.gpu_id}"):
        video_frames = extract_frames(sample["image"]) # video
        image_tensors = process_images(video_frames, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensors]
        image_sizes = [frame.size for frame in video_frames]

        convs = sample["conversations"]
        des_questions, des_answers_gt = convs[0]["value"], convs[1]["value"]
        bbox_questions, bbox_answers_gt = convs[2]["value"], convs[3]["value"]
        act_questions, act_answers_gt = convs[4]["value"], convs[5]["value"]

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], des_questions)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        des_answer_pred = generate_text(model, tokenizer, prompt_question, image_tensors, image_sizes, device=device)

        des_bleu4 = calculate_bleu4(des_answers_gt, des_answer_pred)
        des_rouge_l = calculate_rouge_l(des_answers_gt, des_answer_pred)
        des_meteor = calculate_meteor(des_answers_gt, des_answer_pred)

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], des_questions)
        conv.append_message(conv.roles[1], des_answer_pred)
        conv.append_message(conv.roles[0], bbox_questions)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        bbox_answers_pred = generate_text(model, tokenizer, prompt_question, image_tensors, image_sizes, device=device)

        bbox_bleu4 = calculate_bleu4(bbox_answers_gt, bbox_answers_pred)
        bbox_rouge_l = calculate_rouge_l(bbox_answers_gt, bbox_answers_pred)
        bbox_meteor = calculate_meteor(bbox_answers_gt, bbox_answers_pred)

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], des_questions)
        conv.append_message(conv.roles[1], des_answer_pred)
        conv.append_message(conv.roles[0], bbox_questions)
        conv.append_message(conv.roles[1], bbox_answers_pred)
        conv.append_message(conv.roles[0], act_questions)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        act_answer_pred = generate_text(model, tokenizer, prompt_question, image_tensors, image_sizes, device=device)

        act_bleu4 = calculate_bleu4(act_answers_gt, act_answer_pred)
        act_rouge_l = calculate_rouge_l(act_answers_gt, act_answer_pred)
        act_meteor = calculate_meteor(act_answers_gt, act_answer_pred)

        true_positive, false_positive, false_negative = calculate_precision_recall(act_answers_gt, act_answer_pred)
        
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative


        if is_b2d:
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], des_questions)
            conv.append_message(conv.roles[1], des_answer_pred)
            conv.append_message(conv.roles[0], bbox_questions)
            conv.append_message(conv.roles[1], bbox_answers_pred)
            conv.append_message(conv.roles[0], act_questions)
            conv.append_message(conv.roles[1], act_answer_pred)
            prompt_question = conv.get_prompt()

            action_gt = sample["action"]
            intput_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            action_gt = torch.tensor(action_gt, dtype=torch.float16, device=device).unsqueeze(0)
            (intput_ids, position_ids, attention_mask, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(intput_ids, None, None, None, None, image_tensors, modalities="image", image_sizes=image_sizes) # or video
            hidden_states = model.model(input_ids=intput_ids, position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,output_hidden_states=True).hidden_states[-1]

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden_states = hidden_states * extended_attention_mask
                sum_hidden_states = torch.sum(masked_hidden_states, dim=1, keepdim=True)
                sum_attention_mask = torch.sum(extended_attention_mask, dim=1, keepdim=True)
                average_hidden_states = sum_hidden_states / sum_attention_mask
            else:
                average_hidden_states = torch.mean(hidden_states, dim=1, keepdim=True)
            _, action_pred = model.decode_aeb(average_hidden_states, action_gt)

            conversations.append({
                "id": sample["id"],
                "image": sample["image"], # video
                "description_question": des_questions,
                "description_answer_gt": des_answers_gt,
                "description_answer_pred": des_answer_pred,
                "bbox_question": bbox_questions,
                "bbox_answer_gt": bbox_answers_gt,
                "bbox_answer_pred": bbox_answers_pred,
                "action_question": act_questions,
                "action_answer_gt": act_answers_gt,
                "action_answer_pred": act_answer_pred,
                "action_gt": action_gt[0].detach().cpu().numpy()[1].item(),
                "action_pred": action_pred.detach().cpu().numpy()[0].item()
            })
        else:
            conversations.append({
                "id": sample["id"],
                "image": sample["image"], # video
                "description_question": des_questions,
                "description_answer_gt": des_answers_gt,
                "description_answer_pred": des_answer_pred,
                "bbox_question": bbox_questions,
                "bbox_answer_gt": bbox_answers_gt,
                "bbox_answer_pred": bbox_answers_pred,
                "action_question": act_questions,
                "action_answer_gt": act_answers_gt,
                "action_answer_pred": act_answer_pred
            })
        results.append({
            "id": sample["id"],
            "description_bleu4": des_bleu4,
            "description_rouge_l": des_rouge_l,
            "description_meteor": des_meteor,
            "bbox_bleu4": bbox_bleu4,
            "bbox_rouge_l": bbox_rouge_l,
            "bbox_meteor": bbox_meteor,
            "action_bleu4": act_bleu4,
            "action_rouge_l": act_rouge_l,
            "action_meteor": act_meteor,
        })
        del image_tensors
        del hidden_states
        del intput_ids
        torch.cuda.empty_cache()
        gc.collect()
        
    overall_precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0.0
    overall_recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0.0

    results.append({
        "id": "overall",
        "description_bleu4": None,
        "description_rouge_l": None,
        "description_meteor": None,
        "bbox_bleu4": None,
        "bbox_rouge_l": None,
        "bbox_meteor": None,
        "action_bleu4": None,
        "action_rouge_l": None,
        "action_meteor": None,
        "action_precision": overall_precision,
        "action_recall": overall_recall
    })
    save_conversations(conversations, output_path=f"./exp/{args.exp_name}/gpu_{args.num}_conversations.json")
    save_results_as_table(results, filename=f"./exp/{args.exp_name}/gpu_{args.num}_evaluation_results.csv")

    # release memory
    del model
    torch.cuda.empty_cache()
if __name__ == "__main__":
    args = parse_args()
    device = "cuda"
    
    main()