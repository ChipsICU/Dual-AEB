import json
mm_au_data = json.load(open("LOTVS-MM-AU/mm-au_metadata.json", "r"))
mm_au_frame_data = json.load(open("LOTVS-MM-AU/dataset/mm_au_frame_data.json", "r"))

mm_au_frame_data_result = {}
image_id_to_name = {image['id']: image['file_name'].split(".")[0] for image in mm_au_frame_data['images']}
anno_category = {cat['id']: cat['name'] for cat in mm_au_frame_data['categories']}
for annotation in mm_au_frame_data['annotations']:
    image_name = image_id_to_name[annotation['image_id']]
    category_name = anno_category[annotation['category_id']] if annotation['category_id'] in anno_category else "van"
    bbox = str([round(coord, 2) for coord in annotation['bbox']])
    if image_name in mm_au_frame_data_result:
        mm_au_frame_data_result[image_name].append({"bbox": bbox, "category_name": category_name})
    else:
        mm_au_frame_data_result[image_name] = [{"bbox": bbox, "category_name": category_name}]


import os
def get_frame_info(video_path, frame_id):
    frame_name = video_path.split(os.sep)
    frame_name = f"{frame_name[-3]}_{frame_name[-2]}"
    if "DADA-DATA" in video_path:
        frame_id = str(frame_id).zfill(4)
        frame_path = os.path.join(video_path, frame_id+".png")
    elif "CAP-DATA" in video_path:
        frame_id = str(frame_id).zfill(6)
        frame_path = os.path.join(video_path, frame_id+".jpg")
    frame_name = f"{frame_name}_{frame_id}"
    # assert os.path.exists(frame_path), f"frame path {frame_path} not exist!"
    if not os.path.exists(frame_path):
        print(f"frame path {frame_path} not exist!")
        return None, None
    return frame_name, frame_path

import random

def get_environment_des(meta_data):
    weather_map = {str(i): weather for i, weather in enumerate(["sunny", "rainy", "snowy", "foggy"], 1)}
    light_map = {str(i): light for i, light in enumerate(["day", "night"], 1)}
    scenes_map = {str(i): scene for i, scene in enumerate(["highway", "tunnel", "mountain", "urban", "rural"], 1)}
    linear_map = {str(i): linear for i, linear in enumerate(["arterials", "curve", "intersection", "T-junction", "ramp"], 1)}

    # ----> get environment information
    weather = weather_map[meta_data['weather']]
    light = light_map[meta_data['light']]
    scene = scenes_map[meta_data['scenes']]
    linear = linear_map[meta_data['linear']]

    templates = [
        "The ego car is driving on a {weather} {light} in a {scene} area on a {linear} road.",
        "The ego car is navigating through a {weather} {light} on a {scene} {linear} road.",
        "The vehicle is moving along a {linear} road under {weather} conditions during the {light} in a {scene} setting.",
        "On a {weather} {light}, the ego vehicle travels through a {scene} area along a {linear} roadway.",
        "The car drives along a {linear} road under {weather} skies during the {light}, situated in a {scene} environment.",
        "Navigating a {linear} road in {scene}, the car encounters {weather} conditions during the {light}.",
        "The journey takes place on a {linear} road in a {scene} area under {weather} conditions during the {light}.",
        "The ego vehicle is operating on a {linear} road with {weather} weather in a {scene} area during the {light}.",
        "Under {weather} conditions and {light} light, the car is driving on a {linear} road in a {scene} setting.",
        "The ego car progresses along a {linear} road through a {scene} environment in {weather} weather during the {light}."
    ]

    # Randomly select one template
    selected_template = random.choice(templates)
    
    # Return the formatted description
    return selected_template.format(weather=weather, light=light, scene=scene, linear=linear)

def get_object_templates(object_list):
    templates = {
        'motorcycle': [
            "There is a motorcycle on the road. Be cautious of its movements, especially during lane changes or turns.",
            "A motorcycle is nearby. Keep an eye on it, as motorcyclists may make sudden maneuvers.",
            "Watch out for the motorcycle on the road. Ensure you give it enough space to maneuver safely.",
            "A motorcycle is present in the traffic. Stay alert, particularly when it is changing lanes or making turns.",
            "There is a motorcycle in your vicinity. Maintain a safe distance and be mindful of its movements."
        ],
        'truck': [
            "There is a truck ahead. Maintain a safe distance, as trucks require more space to stop and may have larger blind spots.",
            "A truck is on the road ahead. Be aware of its longer stopping distances and limited visibility.",
            "Watch for the truck in front. Keep your distance, as trucks have significant blind spots and longer braking times.",
            "A truck is nearby. Ensure you keep a safe distance and be cautious of its blind spots.",
            "There is a truck on the road. It may have difficulty stopping quickly, so stay back and be alert."
        ],
        'bus': [
            "There is a bus in the area. Be aware of frequent stops and passengers boarding or alighting.",
            "A bus is nearby. Expect it to make frequent stops, and watch out for pedestrians.",
            "Watch for the bus ahead. It may stop suddenly to pick up or drop off passengers.",
            "A bus is on the road. Be cautious, especially near bus stops where pedestrians might be crossing.",
            "There is a bus in your vicinity. Stay alert for sudden stops and pedestrians entering or exiting the bus."
        ],
        'traffic light': [
            "There is a traffic light ahead. Follow its signals carefully to avoid collisions.",
            "A traffic light is approaching. Make sure to adhere to its signals to ensure safe driving.",
            "Watch for the traffic light ahead. Obey the signals to prevent any accidents.",
            "A traffic light is visible up ahead. Ensure you comply with its signals to maintain safety.",
            "There is a traffic light in your path. Follow the signals closely to navigate the intersection safely."
        ],
        'person': [
            "There is a person nearby. Slow down and be prepared to stop if they enter the roadway.",
            "A pedestrian is present. Reduce your speed and be ready to yield if necessary.",
            "Watch for the person on or near the road. Slow down and be vigilant for any sudden movements.",
            "There is a pedestrian in the area. Be cautious and prepared to stop if they move onto the road.",
            "A person is walking nearby. Maintain a low speed and be ready to brake if they approach the roadway."
        ],
        'bicycle': [
            "There is a bicycle on the road. Give the cyclist enough space and watch for sudden movements.",
            "A cyclist is riding nearby. Keep a safe distance and be cautious of any quick changes in direction.",
            "Watch out for the bicycle ahead. Ensure you allow enough room for the cyclist to maneuver safely.",
            "A bicycle is present on the road. Be mindful of the cyclist and provide ample space to avoid collisions.",
            "There is a cyclist in your vicinity. Stay alert for any sudden turns or stops by the bicycle."
        ],
        'car': [
            "There is a car ahead. Maintain a safe following distance and be alert for any changes in speed or direction.",
            "A car is driving in front of you. Keep your distance and watch for any sudden stops or lane changes.",
            "Watch for the car ahead. Stay back and be prepared for any unexpected movements.",
            "A car is in your lane. Ensure you maintain a safe distance and be ready for any sudden changes.",
            "There is a vehicle ahead. Be cautious of its speed and direction, and keep a safe distance."
        ],
        'van': [
            "There is a van ahead. Maintain a safe distance and be mindful of its blind spots.",
            "A van is on the road ahead. Keep back and be aware that it may have limited visibility.",
            "Watch out for the van in front. Ensure you maintain a safe distance and stay out of its blind spots.",
            "A van is driving nearby. Keep your distance and be alert for any unexpected stops or turns.",
            "There is a van on the road. Be cautious, as vans may have larger blind spots and longer stopping distances."
        ]
    }
    
    result = ""
    
    for obj in object_list[:3]:
        category = obj['category_name']
        bbox = obj['bbox']
        if category in templates:
            template = random.choice(templates[category]) + f" The corresponding bounding box is {bbox}."
            result += template + " \n"    
    return result

def get_decision_res(meta_data, des_type="normal"):
    
    # ----> get decision templates
    decision_templates = {
        "normal": [
            "Maintain normal driving behavior.",
            "Continue driving steadily.",
            "Proceed with regular driving.",
            "Drive normally.",
            "Keep driving as usual."
        ],
        "early warning": [
            "Stay vigilant, as conditions may change quickly.",
            "Slow down a bit and stay extra alert for any potential risks on the road.",
            "Reduce your speed and heighten your attention to possible dangers ahead.",
            "Decrease your speed slightly and keep an eye out for any emerging threats.",
            "Stay cautious, as early signs of danger are present."
        ],
        "emergency braking": [
            "Prepare for emergency braking",
            "Get ready to brake urgently.",
            "Brace for an emergency stop.",
            "Emergency braking may be required.",
            "Prepare to stop suddenly."
        ]
    }

    
    # ----> get accident description
    texts = meta_data['texts']
    causes = meta_data['causes']
    measures = meta_data['measures']
    
    # Templates for reasoning and environment safety
    safety_templates = [
        "The current surroundings appear to be secure.",
        "No immediate threats are detected.",
        "The environment seems stable for now.",
        "No visible hazards are present.",
        "The area is currently free of noticeable dangers.",
        "No significant risks are evident at the moment."
    ]
    
    caution_templates = [
        "But you need to be careful of the situation of {text}.",
        "Pay attention to the situation of {text}.",
        "However, remain cautious of the {text}.",
        "Still, keep an eye on the {text}.",
        "Nevertheless, be aware of the {text}.",
        "Ensure you're vigilant about the {text}."
    ]
    
    reasoning_phrases = [
        "especially if",
        "particularly when",
        "in case",
        "notably if",
        "specifically when",
        "chiefly if"
    ]
    
    # Start constructing the decision response
    decision = des_type[0].upper() + des_type[1:] + ". " + random.choice(decision_templates[des_type])

    if des_type == "emergency braking":
        # Add reasoning using the causes
        reasoning = " " + random.choice(reasoning_phrases) + " " + causes + "." + " " + measures
        decision += reasoning
    else:
        # Add a safety message and caution
        safety_message = random.choice(safety_templates)
        caution_message = random.choice(caution_templates).format(text=texts)
        decision += " " + safety_message + " " + caution_message
    return decision

def question_template(question_type):
    # ----> scene description question templates, answer will be weather, scene, light, linear, etc.
    q1_templates = [
        "<image>\nThis is a scene driving video of the ego car, describing the environment in the video.",
        "<image>\nDescribe the environmental conditions in the video where the ego car is driving.",
        "<image>\nWhat are the environmental details captured in this scene driving video?",
        "<image>\nExplain the surroundings and conditions in this driving scene.",
        "<image>\nProvide a description of the environment where the ego car is operating."
    ]

    # ----> driving key objects question templates, answer will be the objects detected in the video.
    q2_templates = [
        "What should I be aware of while driving in this area?",
        "What elements on the road might influence how I drive in this situation?",
        "What factors on the road should I be cautious of while driving?",
        "Which objects or elements in the environment require attention while driving?",
        "Identify the key objects or factors that could impact driving in this scene.",
        "What are the critical elements I should focus on while navigating this area?"
    ]

    # ----> decision question templates, answer will be the decision (including early warning, emergency braking, normal) made by the ego car.
    q3_templates = [
        "<aeb>\nConsidering the current driving conditions, what should I do?",
        "<aeb>\nConsidering the current driving conditions, what should be my next move and why?",
        "<aeb>\nWhat should I do next while driving in this situation?",
        "<aeb>\nWhat is the appropriate action for me to take next?",
        "<aeb>\nWhat should I do next while driving, and why?",
        "<aeb>\nGiven the driving scenario, what is the best course of action?",
        "<aeb>\nBased on the present conditions, what driving decision should I make?",
        "<aeb>\nWhat is the recommended driving action for this situation?",
        "<aeb>\nWhat steps should I take while driving in the current environment?",
        "<aeb>\nWhat is the best driving decision considering the present conditions?"
    ]
    
    # # ----> accident avoid question templates, answer will be the scene-centric methods to avoid the accident.
    # q4_templates = [
    #     "What actions can be taken to prevent an accident in this scenario?",
    #     "What measures can be implemented to avoid a collision in this situation?",
    #     "How can an accident be averted in this driving scenario?",
    #     "What steps can be taken to prevent the abnormalities in this environment?",
    # ]

    # Mapping the question type to the corresponding templates
    templates_map = {
        "scene_description": q1_templates,
        "driving_key_objects": q2_templates,
        "decision": q3_templates,
        # "method": q4_templates
    }

    # Return a randomly selected template from the corresponding list
    return random.choice(templates_map.get(question_type, []))

# Action_type has three type, normal, early warning, emergency braking
''' mm_au data structure
weather: sunny,rainy,snowy,foggy (1-4)
light: day,night (1-2)
scenes: highway,tunnel,mountain,urban,rural (1-5)
linear: arterials,curve,intersection,T-junction,ramp (1-5)
accident occurred: whether an accident occurred (1/0)
t_ai: Accident window start frame
t_co: Collision start frame
t_ae: Accident window end frame
texts: Description of the accident
causes: Causes of the accident
measures: Advice on how to avoid accident
'''
mm_au_aeb_data_index = {"normal": [], "early warning": [], "emergency braking": []}
window_size = 10 # sliding window size
step_size = 10 # sliding window step size
min_frame_distance = 5 # minimum frame distance between the accident and the action window

for video_key in mm_au_data.keys():

    meta_data = mm_au_data[video_key]
    video_path = meta_data['video_path']

    # ----> get accident information
    acc_occurred = eval(meta_data['accident occurred'])
    t_ai = eval(meta_data['t_ai'])
    t_co = eval(meta_data['t_co'])
    t_ae = eval(meta_data['t_ae'])
    total_frames = eval(meta_data['total_frames'])
    
    # Define the frame ranges for each action type using sliding windows
    if acc_occurred == 1:
        # Emergency braking
        for start in range(max(1, t_co - window_size + 1), t_co + 5, step_size):
            end = start + window_size - 1
            if min(end, t_co + 5, total_frames) - start >= min_frame_distance:
                # Generate environment description
                description = get_environment_des(meta_data)
                decision = get_decision_res(meta_data, des_type="emergency braking")
                mm_au_aeb_data_index["emergency braking"].append((video_path, start, min(end, t_co + 5, total_frames), description, decision))
        
        # Early warning
        for start in range(max(1, t_ai - window_size + 1), min(t_ai + window_size, t_co - window_size + 1), step_size):
            end = start + window_size - 1
            if min(end, t_co - window_size + 1) - start >= min_frame_distance:
                # Generate environment description
                description = get_environment_des(meta_data)
                decision = get_decision_res(meta_data, des_type="early warning")
                mm_au_aeb_data_index["early warning"].append((video_path, start, min(end, t_co - window_size + 1), description, decision))
                break
        
        # Normal
        for start in range(max(1, t_ai - window_size * 2), t_ai - window_size, step_size):
            end = start + window_size - 1
            if end - start >= min_frame_distance:
                # Generate environment description
                description = get_environment_des(meta_data)
                decision = get_decision_res(meta_data, des_type="normal")
                mm_au_aeb_data_index["normal"].append((video_path, start, end, description, decision))

    else:
        # No accident, so only early warning and normal
        # Early warning
        for start in range(max(1, t_ai - window_size + 1), min(t_ai + window_size, t_co - window_size + 1), step_size):
            end = start + window_size - 1
            if min(end, t_co - window_size + 1) - start >= min_frame_distance:
                # Generate environment description
                description = get_environment_des(meta_data)
                decision = get_decision_res(meta_data, des_type="early warning")
                mm_au_aeb_data_index["early warning"].append((video_path, start, min(end, t_co - window_size + 1), description, decision))
                break
        
        # Normal
        for start in range(max(1, t_ai - window_size * 2), t_ai - window_size, step_size):
            end = start + window_size - 1
            if end - start >= min_frame_distance:
                # Generate environment description
                description = get_environment_des(meta_data)
                decision = get_decision_res(meta_data, des_type="normal")
                mm_au_aeb_data_index["normal"].append((video_path, start, end, description, decision))

import hashlib

def get_hash_key(video_path, action_type, idx):
    key = f"{video_path}_{action_type}_{idx}"
    encoded_data = str(key).encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    return hash_hex

def get_video_path_list(video_path, start_frame, end_frame):
    video_path_list = []
    for frame_id in range(start_frame, end_frame):
        _, frame_path = get_frame_info(video_path, frame_id)
        video_path_list.append(frame_path)
    return video_path_list


mm_au_aeb_data = []
idx = 0
for action_type in mm_au_aeb_data_index.keys():
    for video_path, start_frame, end_frame, description, decision in mm_au_aeb_data_index[action_type]:
        end_frame_id, _ = get_frame_info(video_path, end_frame)
        
        if end_frame_id is None:
            continue
        if end_frame_id not in mm_au_frame_data:
            print(f"frame {end_frame_id} not in the frame data, skip!")
            continue
        end_frame_anno = mm_au_frame_data[end_frame_id] 

        video_path_list = get_video_path_list(video_path, start_frame, end_frame)
        assert None not in video_path_list, f"video path list {video_path_list} contains None!"
        
        scene_question, key_element_question, decision_question = question_template("scene_description"), question_template("driving_key_objects"), question_template("decision")

        key_element_answer = get_object_templates(end_frame_anno)
        frame = end_frame
        while key_element_answer == "" and frame > start_frame:
            frame -= 1
            frame_id, _ = get_frame_info(video_path, frame)
            frame_anno = mm_au_frame_data[frame_id]
            key_element_answer = get_object_templates(frame_anno)
        
        if key_element_answer == "":
            # print(f"key element answer is empty, skip!")
            continue

        id = get_hash_key(video_path, action_type, idx)

        conversation = [{"from": "human", "value": scene_question}, {"from": "gpt", "value": description}, \
                        {"from": "human", "value": key_element_question}, {"from": "gpt", "value": key_element_answer}, \
                        {"from": "human", "value": decision_question}, {"from": "gpt", "value": decision}]
        
        # if measures is not None:
        #     conversation.extend([{"from": "human", "value": measures_question}, {"from": "gpt", "value": measures}])
        
        mm_au_aeb_data.append({
            "id": id,
            "video": video_path_list,
            "conversations": conversation,
        })

        idx += 1
