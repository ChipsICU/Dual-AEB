import numpy as np
import cv2
import math
import os
import re
import random
import os
import gzip
import json
import hashlib

import webcolors

def get_colour_name(rgb_triplet):
    # full list: https://www.w3schools.com/tags/ref_colornames.asp
    myColors = {
        "red"         : "#ff0000", # R
        "orange"      : "#ffa500", # O
        "yellow"      : "#ffff00", # Y
        "green"       : "#008000", # G
        "blue"        : "#0000ff", # B
        "magenta"     : "#ff00ff", # I
        "purple"      : "#800080", # V
        
        "coral"       : "#ff7f50", # light red
        "maroon"      : "#800000", # dark red
        "navy"        : "#000080", # dark blue
        "cyan"        : "#00ffff", # light blue
        "gold"        : "#ffd700", # dark yellow
        "lime"        : "#00ff00", # bright green
        "jade"        : "#00a36c", # light green
        "olive"       : "#808000", # dark green
        "grey"        : "#808080", # grey

        "black"       : "#000000", # black
        "white"       : "#ffffff", # white
        "silver"      : "#c0c0c0", # silver
        "teal"        : "#008080", # teal
        "indigo"      : "#4b0082", # indigo
        "violet"      : "#ee82ee", # violet
        "pink"        : "#ffc0cb", # pink
        "brown"       : "#a52a2a", # brown
        "beige"       : "#f5f5dc", # beige
        "khaki"       : "#f0e68c", # khaki
        "peach"       : "#ffe5b4", # peach
        "lavender"    : "#e6e6fa", # lavender
        "turquoise"   : "#40e0d0", # turquoise
        "salmon"      : "#fa8072", # salmon
        "crimson"     : "#dc143c", # crimson
        "plum"        : "#dda0dd", # plum
        "orchid"      : "#da70d6", # orchid
        "chartreuse"  : "#7fff00", # chartreuse
        "aquamarine"  : "#7fffd4", # aquamarine
        "sienna"      : "#a0522d", # sienna
        "chocolate"   : "#d2691e", # chocolate
        "tan"         : "#d2b48c", # tan
        "wheat"       : "#f5deb3", # wheat
        "peru"        : "#cd853f", # peru
        "slategray"   : "#708090", # slate gray
        "lightgray"   : "#d3d3d3", # light gray
        "darkgray"    : "#a9a9a9", # dark gray
        "dodgerblue"  : "#1e90ff", # dodger blue
        "royalblue"   : "#4169e1", # royal blue
        "skyblue"     : "#87ceeb", # sky blue
        "steelblue"   : "#4682b4", # steel blue
        "goldenrod"   : "#daa520", # goldenrod
        "darkgoldenrod": "#b8860b", # dark goldenrod
        "lightgreen"  : "#90ee90", # light green
        "forestgreen" : "#228b22", # forest green
        "darkolivegreen": "#556b2f", # dark olive green
        "seagreen"    : "#2e8b57", # sea green
        "darkslategray": "#2f4f4f", # dark slate gray
        "midnightblue": "#191970", # midnight blue
        "royalblue"   : "#4169e1", # royal blue
        "mediumblue"  : "#0000cd", # medium blue
        "darkblue"    : "#00008b", # dark blue
        "mediumvioletred": "#c71585", # medium violet red
        "palevioletred": "#db7093", # pale violet red
        "mediumorchid": "#ba55d3", # medium orchid
        "mediumpurple": "#9370db", # medium purple
        "blueviolet"  : "#8a2be2", # blue violet
        "darkviolet"  : "#9400d3", # dark violet
        "darkorchid"  : "#9932cc", # dark orchid
        "darkmagenta" : "#8b008b", # dark magenta
        "deeppink"    : "#ff1493", # deep pink
        "lightpink"   : "#ffb6c1", # light pink
        "hotpink"     : "#ff69b4", # hot pink
        "indianred"   : "#cd5c5c", # indian red
        "firebrick"   : "#b22222", # firebrick
        "darkred"     : "#8b0000", # dark red
        "snow"        : "#fffafa", # snow white
        "floralwhite" : "#fffaf0", # floral white
        "ivory"       : "#fffff0", # ivory white
        "seashell"    : "#fff5ee", # seashell white
        "honeydew"    : "#f0fff0", # honeydew
        "mintcream"   : "#f5fffa", # mint cream
        "azure"       : "#f0ffff", # azure
        "aliceblue"   : "#f0f8ff", # alice blue
        "ghostwhite"  : "#f8f8ff", # ghost white
        "whitesmoke"  : "#f5f5f5", # white smoke
        "gainsboro"   : "#dcdcdc", # gainsboro
        "linen"       : "#faf0e6", # linen
        "oldlace"     : "#fdf5e6", # old lace
        "papayawhip"  : "#ffefd5", # papaya whip
        "blanchedalmond": "#ffebcd", # blanched almond
        "bisque"      : "#ffe4c4", # bisque
        "antiquewhite": "#faebd7", # antique white
        "burlywood"   : "#deb887", # burlywood
        "navajowhite" : "#ffdead", # navajo white
        "tan"         : "#d2b48c", # tan
        "rosybrown"   : "#bc8f8f", # rosy brown
        "moccasin"    : "#ffe4b5", # moccasin
        "sandybrown"  : "#f4a460", # sandy brown
        "wheat"       : "#f5deb3", # wheat
        "beige"       : "#f5f5dc", # beige
        "chocolate"   : "#d2691e", # chocolate
        "peru"        : "#cd853f", # peru
        "saddlebrown" : "#8b4513", # saddle brown
        "sienna"      : "#a0522d", # sienna
        "brown"       : "#a52a2a", # brown
        "darkorange"  : "#ff8c00", # dark orange
        "orangered"   : "#ff4500", # orange red
        "tomato"      : "#ff6347", # tomato
        "darkkhaki"   : "#bdb76b", # dark khaki
        "palegoldenrod": "#eee8aa", # pale goldenrod
        "lemonchiffon": "#fffacd", # lemon chiffon
        "lightgoldenrodyellow": "#fafad2", # light goldenrod yellow
        "lightyellow" : "#ffffe0", # light yellow
        "lightcyan"   : "#e0ffff", # light cyan
        "paleturquoise": "#afeeee", # pale turquoise
        "aqua"        : "#00ffff", # aqua
        "aquamarine"  : "#7fffd4", # aquamarine
        "mediumaquamarine": "#66cdaa", # medium aquamarine
        "mediumseagreen": "#3cb371", # medium sea green
        "seagreen"    : "#2e8b57", # sea green
        "darkgreen"   : "#006400", # dark green
        "darkcyan"    : "#008b8b", # dark cyan
        "darkslategray": "#2f4f4f", # dark slate gray
        "midnightblue": "#191970", # midnight blue
        "cornflowerblue": "#6495ed", # cornflower blue
        "deepskyblue" : "#00bfff", # deep sky blue
        "lightskyblue": "#87cefa", # light sky blue
        "slateblue"   : "#6a5acd", # slate blue
        "mediumslateblue": "#7b68ee", # medium slate blue
        "darkslateblue": "#483d8b", # dark slate blue
        "lavenderblush": "#fff0f5", # lavender blush
        "mistyrose"   : "#ffe4e1"  # misty rose
    }

        
    min_colours = {}
    for name, hex_val in myColors.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    
    return min_colours[min(min_colours.keys())]

def parse_rgb_string(rgb_string):
    return tuple(map(int, rgb_string.split(',')))

WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

DIS_CAR_SAVE = 100
DIS_WALKER_SAVE = 100
DIS_SIGN_SAVE = 100
DIS_LIGHT_SAVE = 100

edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

traffic_light_colors = {0: "RED", 1: "YELLOW", 2: "GREEN", 3: "OFF"}

traffic_sign_types = [
    "traffic.speed_limit.30",
    "traffic.speed_limit.40",
    "traffic.speed_limit.50",
    "traffic.speed_limit.60",
    "traffic.speed_limit.90",
    "traffic.speed_limit.100",
    "traffic.stop",
    "traffic.yield",
    "traffic.no_entry",
    "traffic.pedestrian_crossing",
    "traffic.light"
]
def generate_formal_language_mapping(sign_type):
    if "speed_limit" in sign_type:
        speed = sign_type.split('.')[-1]
        return f"a speed limit of {speed} km/h"
    elif "stop" in sign_type:
        return "a stop sign"
    elif "yield" in sign_type:
        return "a yield sign"
    elif "no_entry" in sign_type:
        return "a no entry sign"
    elif "pedestrian_crossing" in sign_type:
        return "a pedestrian crossing"
    elif "light" in sign_type:
        return "a traffic light"
    else:
        return "an unknown traffic sign"
    
traffic_sign_mapping = {sign: generate_formal_language_mapping(sign) for sign in traffic_sign_types}


# Skeleton Weather Templates
weather_templates = {"0": ["The sky is almost clear with plenty of sunshine, and the wind is mild. There is no rain or signs of wetness, the air is fresh, and visibility is high with very light fog. Enjoy the clear weather, but always stay alert. Even in good conditions, it's important to keep a safe distance from other vehicles.", "Clear skies with minimal cloud cover, bright sunlight, and a gentle breeze. Visibility is excellent, with only a slight haze. Ideal driving conditions. However, remain cautious and keep an eye on speed limits and other road users."],
"1":["The sky is mostly clear, with the sun just rising. The wind is light, and the air is fresh, with high visibility and very light fog. Be aware of changing light conditions as the sun rises. Early morning drivers should be mindful of glare and stay vigilant.", "Clear skies at dawn with gentle winds. The air is crisp, but watch for any potential sun glare as visibility improves. Morning driving can be tricky with sun glare, so make sure to use your sun visor and keep your windshield clean."],
"2":["The sky is partly cloudy with moderate rain. The wind is picking up, and the ground is getting wet, with fog starting to thicken. Drive cautiously in wet conditions. Keep a safe distance from the car in front, and watch out for slick spots on the road.", "Light clouds cover the sky, accompanied by moderate rain and stronger winds. The ground is becoming damp, and visibility may be slightly reduced due to fog. Wet roads can be dangerous. Reduce your speed, use your headlights, and be prepared for sudden stops."],
"3":["Dark clouds fill the sky, with heavy rain and strong winds. The ground is slick, visibility is low, and the air is filled with thick fog. Extreme caution is required in these conditions. Slow down, increase your following distance, and be prepared for potential hazards on the road.", "Heavy rain and strong winds are battering the area, with thick clouds overhead. Visibility is poor due to the dense fog. Proceed with extreme caution, use your low beams, and keep both hands on the wheel."],
"5":["The sky is overcast, but there is no rain. The wind is moderate, and the air is humid. The ground is slightly wet, with noticeable fog. Even without rain, wet roads can be slippery. Drive carefully, and be mindful of potential hydroplaning in damp conditions.", "Thick clouds cover the sky with no rain, but the wind is steady, and the air feels moist. Visibility is reduced due to the fog. Maintain a steady speed and be extra cautious on curves and intersections where visibility is limited."],
"6":["The sky is mostly clear, with light winds. The air is fresh, but fog is significantly reducing visibility.", "Use fog lights if you have them, and drive slowly. Keep a greater distance from the vehicle in front of you.", "Clear skies with very light wind, but heavy fog is present, making it difficult to see far ahead. Visibility is crucial; drive with caution and avoid overtaking unless absolutely necessary." ],
"7":["The sky is clear with plenty of sunlight and light winds. The air is fresh, with only a slight haze. Perfect conditions for driving, but stay alert for any sudden changes in weather or road conditions.", "Clear skies and mild winds make for excellent driving conditions. Visibility is good, with minimal fog. Enjoy the smooth ride, but don’t get too complacent. Always be prepared for other drivers who may not be as attentive."],
"8":["The sky is covered with thick clouds, with heavy rain and strong winds. The ground is slick, and visibility is extremely low due to dense fog. Proceed very slowly, use your low beams, and increase your following distance significantly.", ],
"9":["The sky is cloudy with moderate to heavy rain and strong winds. The ground is wet, and visibility is reduced due to thick fog. Keep a safe distance from other vehicles and reduce speed. Be aware of possible water buildup on the road, which could cause hydroplaning.", "Cloudy skies with heavy rain and strong winds. The road is wet and slippery, with fog making it difficult to see far ahead. Slow down and drive with caution. Use your headlights and be extra careful when approaching intersections or sharp turns."],
"10":["Partly cloudy skies with heavy rain and strong winds. The ground is wet, and fog is reducing visibility. Drive cautiously in these conditions. Use your headlights, slow down, and be prepared for sudden stops.", "Clouds are covering the sky, with rain pouring down and wind blowing hard. Fog is present, making driving more challenging. It’s essential to reduce speed and maintain a greater distance from the vehicle ahead. Stay alert for any changes in the weather."],
"11":["The sky is dark with thick clouds, and moderate rain is falling with strong winds. The ground is slippery, and visibility is poor due to heavy fog. In such conditions, slow down and drive defensively. Use your low beams and avoid sudden maneuvers.", "Thick clouds cover the sky with heavy rain and strong winds. The road is slick, and visibility is low. Exercise caution while driving, especially in areas with poor drainage. Keep your headlights on and be prepared for sudden stops."],
"12":["The sky is mostly clear with plenty of sunlight and mild winds. However, heavy fog is significantly reducing visibility. Use fog lights and reduce your speed. Stay focused and avoid distractions as visibility is low.", "Clear skies with light winds, but the thick fog makes it hard to see ahead. Drive carefully and use your low beams. Keep a safe distance from the car in front, and avoid sudden lane changes."],
"13":["The sky is mostly clear at dawn with light winds. However, heavy fog is making visibility challenging. Early morning fog can be dense; drive slowly and use your headlights to improve visibility.", "Clear skies at sunrise with gentle winds, but thick fog limits visibility. Use caution when driving through fog. Slow down, and keep a greater distance from the vehicle ahead."],
"14":["The sky is covered with dark clouds, with heavy rain, strong winds, and thick fog. The ground is wet and slippery. Avoid driving if possible. If you must drive, proceed slowly, use your headlights, and be extra cautious of flooded roads.", "Torrential rain and powerful winds are accompanied by thick fog, making driving extremely difficult. Extreme caution is needed. Keep a significant distance from other vehicles, and avoid any sudden maneuvers."],
"15":["The sky is overcast with heavy rain, strong winds, and dense fog. The road is completely wet and slippery. Conditions are dangerous. Drive at a reduced speed, use your low beams, and be aware of hydroplaning on wet roads.", "Dark clouds, heavy rain, and powerful winds create hazardous driving conditions, with thick fog reducing visibility. Slow down, turn on your headlights, and be prepared for sudden stops or obstacles on the road."],
"18":["Partly cloudy skies with light winds. The air is fresh, with minimal fog and good visibility. These are generally good driving conditions, but stay alert, especially on winding roads or in areas with sudden weather changes.", "The sky is partly cloudy with a gentle breeze. Visibility is good, though there’s a slight haze. Enjoy the pleasant weather, but remain cautious. Keep an eye out for other drivers and potential hazards on the road."],
"19":["The sky is partly cloudy with light winds, but the sun is low on the horizon, which may cause glare. Fog is minimal, but visibility may be affected. Be cautious of sun glare, especially during sunrise or sunset. Use your visor and drive at a moderate speed.", "Partly cloudy skies with a low sun position and light winds. Visibility is slightly reduced by haze. Watch out for glare from the sun, and adjust your speed accordingly. Keep your windshield clean for better visibility."],
"20":["The sky is covered with thick clouds, moderate rain, and strong winds. The ground is wet, with thick fog reducing visibility. Drive carefully and reduce speed. Keep a safe distance from other vehicles, and use your headlights to improve visibility.", "Overcast skies with moderate rain and strong winds. The road is wet and visibility is limited due to dense fog. Slow down and maintain a safe distance from other cars. Use your low beams and be prepared for potential hazards."],
"21":["The sky is dark with thick clouds, heavy rain, and strong winds. The road is wet, and thick fog makes visibility extremely poor. Avoid driving if possible. Proceed very cautiously, use your low beams, and maintain a significant distance from the vehicle ahead.", "Severe weather with heavy rain, strong winds, and dense fog. The road is extremely slippery, and visibility is almost zero. Drive slowly, and watch for flooded areas"],
"22":["The sky is mostly clear with light winds. However, heavy fog is reducing visibility significantly. Use your fog lights and reduce your speed. Keep a safe distance from the vehicle in front of you.", "Clear skies with minimal wind, but thick fog is making it difficult to see far ahead. Drive cautiously and avoid overtaking. Maintain a safe distance and stay alert for any obstacles on the road."],
"23":["The sky is overcast with moderate rain, strong winds, and thick fog. The road is wet and slippery. Drive slowly and increase your following distance. Use your low beams and be aware of potential hazards, such as hydroplaning.", "Overcast skies with moderate rain and strong winds. The road is slick, and visibility is low due to dense fog. Reduce your speed and drive cautiously. Keep your headlights on and be prepared for sudden changes in road conditions."],
"25": ["The sky is covered with thick clouds, with heavy rain, strong winds, and dense fog. The road is completely wet and slippery. Avoid driving if possible. If you must drive, proceed with extreme caution, use your low beams, and maintain a significant distance from other vehicles.", "Torrential rain and strong winds are making driving conditions hazardous, with dense fog reducing visibility to near zero. Delay travel if possible. If driving is necessary, slow down, use your headlights, and be prepared for flooded areas or sudden stops."],
"26": ["Partly cloudy skies with plenty of sunshine and minimal wind. The air is fresh, and visibility is excellent with little to no fog. Ideal driving conditions, but always remain alert and cautious, especially in areas where the road may suddenly change.", "The sky is partly cloudy with bright sunshine and almost no wind. Visibility is clear, making it a great day for driving. Enjoy the drive, but don’t forget to stay vigilant and watch for other drivers who may not be as attentive."]
}

def get_bounding_boxes(anno):
    # get the bounding boxes of front objects
    cam_map = {
        'CAM_FRONT': 'rgb_front',
        'CAM_FRONT_LEFT': 'rgb_front_left',
        'CAM_FRONT_RIGHT': 'rgb_front_right',
        'CAM_BACK': 'rgb_back', 
        'CAM_BACK_LEFT': 'rgb_back_left', 
        'CAM_BACK_RIGHT': 'rgb_back_right',
        'TOP_DOWN': 'rgb_top_down'
    }
    key = "CAM_FRONT"
    sensors_anno = anno['sensors']
    bounding_boxes = anno['bounding_boxes']

    K = sensors_anno[key]['intrinsic']
    world2cam = sensors_anno[key]['world2cam']
    bbox_list = []

    for npc in bounding_boxes:
        if npc['class'] == 'ego_vehicle': 
            continue
        if npc['distance'] > 35: 
            continue
        if abs(npc['location'][2] - bounding_boxes[0]['location'][2]) > 10: 
            continue
        
        if 'vehicle' in npc['class']: 
            forward_vec = get_forward_vector(sensors_anno[key]['rotation'][2])
            ray = np.array(npc['location']) - np.array(sensors_anno[key]['location'])
            if forward_vec.dot(ray) > 1 and vector_angle(forward_vec, ray) < 45:
                verts = np.array(npc['world_cord'])
                bbox_points = []
                for edge in edges:
                    p1, _ = get_image_point(verts[edge[0]], K, world2cam)
                    p2, _ = get_image_point(verts[edge[1]], K, world2cam)
                    bbox_points.extend([(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))])
                
                if bbox_points:
                    min_x = min(p[0] for p in bbox_points)
                    max_x = max(p[0] for p in bbox_points)
                    min_y = min(p[1] for p in bbox_points)
                    max_y = max(p[1] for p in bbox_points)
                    if "color" in npc.keys():
                        bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance'], 'state': [npc['state'], npc['light_state'], npc['type_id'], get_colour_name(parse_rgb_string(npc['color']))]})
                    else:
                        bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance'], 'state': [npc['state'], npc['light_state'], npc['type_id'], None]})

        else: 
            
            if npc['class'] == 'traffic_sign': 
                npc['extent'][1] = 0.5
            forward_vec = get_forward_vector(sensors_anno[key]['rotation'][2])
            ray = np.array(npc['location']) - np.array(sensors_anno[key]['location'])
            if forward_vec.dot(ray) > 1 and vector_angle(forward_vec, ray) < 45:
                if 'world_cord' in npc.keys():
                    if 'dirtdebris' in npc['type_id']:
                        local_verts = calculate_cube_vertices(npc['bbx_loc'], [npc['extent'][1], npc['extent'][0], npc['extent'][2]])
                        verts = []
                        for l_v in local_verts:
                            g_v = np.dot(np.matrix(npc['world2sign']).I, [l_v[0], l_v[1], l_v[2],1])
                            verts.append(g_v.tolist()[0][:-1])
                    else:
                        verts = np.array(npc['world_cord'])
                else:
                    verts = calculate_cube_vertices(npc['center'], npc['extent'])

                bbox_points = []
                for edge in edges:
                    p1, _ = get_image_point(verts[edge[0]], K, world2cam)
                    p2, _ = get_image_point(verts[edge[1]], K, world2cam)
                    bbox_points.extend([(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))])
                
                if bbox_points:
                    min_x = min(p[0] for p in bbox_points)
                    max_x = max(p[0] for p in bbox_points)
                    min_y = min(p[1] for p in bbox_points)
                    max_y = max(p[1] for p in bbox_points)
                    if npc['class'] == 'traffic_light':
                        bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance'], 'state': traffic_light_colors[npc['state']]})
                    elif npc['class'] == 'traffic_sign':
                        bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance'], 'state': npc['type_id']})
                    elif npc['class'] == 'walker':
                        bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance'], 'state': [npc['gender'], npc['age']]})
                    # bbox_list.append({'class': npc['class'], 'bbox': [(min_x, min_y), (max_x, max_y)], 'distance': npc['distance']})

    bbox_list.sort(key=lambda x: x['distance'], reverse=False)

    return bbox_list


def question_template(question_type):
    # ----> scene description question templates, answer will be weather, scene, light, linear, etc.
    q1_templates = [
        "<image>\nThis is a scene driving video of the ego car, describing the weather in the video.",
        "<image>\nDescribe the weather conditions in the video where the ego car is driving.",
        "<image>\nWhat are the weather details captured in this scene driving video?",
        "<image>\nExplain the surroundings and conditions in this driving scene.",
        "<image>\nProvide a description of the weather where the ego car is operating."
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
        "Considering the current driving conditions, what should I do?",
        "Considering the current driving conditions, what should be my next move and why?",
        "What should I do next while driving in this situation?",
        "What is the appropriate action for me to take next?",
        "What should I do next while driving, and why?",
        "Given the driving scenario, what is the best course of action?",
        "Based on the present conditions, what driving decision should I make?",
        "What is the recommended driving action for this situation?",
        "What steps should I take while driving in the current environment?",
        "What is the best driving decision considering the present conditions?"
    ]
    
    # Mapping the question type to the corresponding templates
    templates_map = {
        "scene_description": q1_templates,
        "driving_key_objects": q2_templates,
        "decision": q3_templates,
    }

    # Return a randomly selected template from the corresponding list
    return random.choice(templates_map.get(question_type, []))

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc[0], loc[1], loc[2], 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    depth = point_camera[2]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    
    return point_img[0:2], depth

def point_in_canvas_wh(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_WIDTH) and (pos[1] >= 0) and (pos[1] < WINDOW_HEIGHT):
        return True
    return False

def get_forward_vector(yaw):
    # Convert the yaw angle from degrees to radians
    yaw_rad = math.radians(yaw)
    # Calculate the X and Y components of the forward vector (in a left-handed coordinate system with Z-axis upwards)
    # Note: In a left-handed coordinate system, the positive Y direction could correspond to either forward or backward, depending on the specific application scenario
    x = math.cos(yaw_rad)
    y = math.sin(yaw_rad)
    # On a horizontal plane, the Z component of the forward vector is 0
    z = 0
    return np.array([x, y, z])

def calculate_cube_vertices(center, extent):
    if isinstance(center, list):
        cx, cy, cz = center
        x, y, z = extent
    else:
        cx, cy, cz = center.x,  center.y,  center.z
        x, y, z = extent.x, extent.y, extent.z
    vertices = [
        (cx + x, cy + y, cz + z),
        (cx + x, cy + y, cz - z),
        (cx + x, cy - y, cz + z),
        (cx + x, cy - y, cz - z),
        (cx - x, cy + y, cz + z),
        (cx - x, cy + y, cz - z),
        (cx - x, cy - y, cz + z),
        (cx - x, cy - y, cz - z)
    ]
    return vertices

def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=5):
    """
    Draw a dashed line on an image.
    Arguments:
    - img: The image on which to draw the dashed line.
    - start_point: The starting point of the dashed line, in the format (x, y).
    - end_point: The ending point of the dashed line, in the format (x, y).
    - color: The color of the dashed line, in the format (B, G, R).
    - thickness: The thickness of the line.
    - dash_length: The length of each dash segment in the dashed line.
    """
    # Calculate total length
    d = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    dx = (end_point[0] - start_point[0]) / d
    dy = (end_point[1] - start_point[1]) / d

    x, y = start_point[0], start_point[1]

    while d >= dash_length:
        # Calculate the end point of the next segment
        x_end = x + dx * dash_length
        y_end = y + dy * dash_length
        cv2.line(img, (int(x), int(y)), (int(x_end), int(y_end)), color, thickness)

        # Update starting point and remaining length
        x = x_end + dx * dash_length
        y = y_end + dy * dash_length
        d -= 2 * dash_length

def world_to_ego(point_world, w2e):
    point_world = np.array([point_world[0], point_world[1], point_world[2], 1])
    point_ego = np.dot(w2e, point_world)
    point_ego = [point_ego[1], -point_ego[0], point_ego[2]]
    return point_ego

def vector_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def get_weather_id(weather_conditions):
    from xml.etree import ElementTree as ET
    tree = ET.parse('./leaderboard/data/weather.xml')
    root = tree.getroot()
    def conditions_match(weather, conditions):
        for (key, value) in weather:
            if key == 'route_percentage' : continue
            if str(conditions[key]) != value:
                return False
        return True
    for case in root.findall('case'):
        weather = case[0].items()
        if conditions_match(weather, weather_conditions):
            return case.items()[0][1]
    return None

def compute_2d_distance(loc1, loc2):
    return math.sqrt((loc1.x-loc2.x)**2+(loc1.y-loc2.y)**2)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def convert_depth(data):
    """
    Computes the normalized depth from a CARLA depth map.
    """
    data = data.astype(np.float16)

    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    normalized /= (256 * 256 * 256 - 1)
    return normalized * 1000

def get_relative_transform(ego_matrix, vehicle_matrix):
    """
    Returns the position of the vehicle matrix in the ego coordinate system.
    :param ego_matrix: ndarray 4x4 Matrix of the ego vehicle in global
    coordinates
    :param vehicle_matrix: ndarray 4x4 Matrix of another actor in global
    coordinates
    :return: ndarray position of the other vehicle in the ego coordinate system
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    relative_pos = rot @ relative_pos

    return relative_pos

def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def build_skeleton(ped, sk_links):
    ######## get the pedestrian skeleton #########
    bones = ped.get_bones()

    # list where we will store the lines we will project
    # onto the camera output
    lines_3d = []

    # cycle through the bone pairs in skeleton.txt and retrieve the joint positions
    for link in sk_links[1:]:

        # get the roots of the two bones to be joined
        bone_transform_1 = next(filter(lambda b: b.name == link[0], bones.bone_transforms), None)
        bone_transform_2 = next(filter(lambda b: b.name == link[1], bones.bone_transforms), None)

        # some bone names aren't matched
        if bone_transform_1 is not None and bone_transform_2 is not None:
            lines_3d.append([(bone_transform_1.world.location.x, bone_transform_1.world.location.y, bone_transform_1.world.location.z), 
                             (bone_transform_2.world.location.x, bone_transform_2.world.location.y, bone_transform_2.world.location.z)]
                            )
    return lines_3d

def get_matrix(location, rotation):
    """
    Creates matrix from carla transform.
    """
    pitch, roll, yaw = rotation
    x, y, z = location
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def parse_file_path(file_path):
    weather_id_pattern = r"Weather(\d+)"
    file_id_pattern = r"/(\d{5})\.json\.gz"

    weather_id_match = re.search(weather_id_pattern, file_path)
    file_id_match = re.search(file_id_pattern, file_path)

    if weather_id_match and file_id_match:
        weather_id = weather_id_match.group(1)
        file_id = file_id_match.group(1)
        return weather_id, file_id
    else:
        return None, None

def generate_object_templates(anno):
    bbox_3d_list = get_bounding_boxes(anno)
    labels_3d_list = ['walker', 'traffic_sign', 'traffic_light', 'vehicle']
    # templates = {
    #     'walker': [
    #         "There is a walker nearby. Reduce speed and be prepared to yield if they step into the roadway.",
    #         "A walker is present. Slow down and be alert for any sudden movements toward the road.",
    #         "Watch for the walker ahead. Maintain a cautious speed and be ready to stop if needed.",
    #         "A walker is close to the road. Be mindful and prepared to take action if they approach your lane.",
    #         "There is a walker in the vicinity. Drive carefully and stay alert for any unexpected actions."
    #     ],
    #     'traffic_sign': [
    #         "There is a traffic sign ahead. Observe the sign and adjust your driving action as required.",
    #         "A traffic sign is visible. Follow the instructions indicated by the sign.",
    #         "Watch for the traffic sign up ahead. Ensure compliance with its guidance for safe driving.",
    #         "A traffic sign is approaching. Pay attention to its message and drive accordingly.",
    #         "There is a traffic sign in your path. Adhere to its instructions to ensure safety on the road."
    #     ],
    #     'traffic_light': [
    #         "There is a traffic light ahead. Follow the light's signals to navigate the intersection safely.",
    #         "A traffic light is coming up. Make sure to adhere to the signals for safe passage.",
    #         "Watch for the traffic light ahead. Obey the signals to avoid any potential hazards.",
    #         "A traffic light is visible up ahead. Ensure you follow the signals closely to maintain safety.",
    #         "There is a traffic light on your path. Comply with the signals to navigate through safely."
    #     ],
    #     'vehicle': [
    #         "There is a vehicle ahead. Maintain a safe following distance and be alert for any changes in speed or direction.",
    #         "A vehicle is driving in front of you. Keep your distance and watch for any sudden stops or lane changes.",
    #         "Watch for the vehicle ahead. Stay back and be prepared for any unexpected movements.",
    #         "A vehicle is in your lane. Ensure you maintain a safe distance and be ready for any sudden changes.",
    #         "There is a vehicle ahead. Be cautious of its speed and direction, and keep a safe distance."
    #     ]
    # }

    result = ""
    seq_template = ["1st", "2nd", "3rd"]
    traffic_light_mapping = {
            "RED": "a red light",
            "YELLOW": "a yellow light",
            "GREEN": "a green light",
            "OFF": "an off signal"
    }
    vehicle_light_mapping = {
        "Brake": "the brake lights are on",
        "RightBlinker": "the right turn signal is blinking",
        "LeftBlinker": "the left turn signal is blinking",
        "Reverse": "the reverse lights are on",
        "NONE": "no significant lights are on"
    }
    for idx, obj in enumerate(bbox_3d_list[:3]):
        category = obj['class']
        distance = obj['distance']
        bbox = obj['bbox']
        special_info = obj.get('state', None)

        if category == "traffic_sign":
            special_info_readable = traffic_sign_mapping.get(special_info, special_info)
            special_info_readable = special_info_readable.replace(".", " ")
            description = f"{category.replace('_', ' ')} displaying {special_info_readable}"
        elif category == "traffic_light":
            special_info_readable = traffic_light_mapping.get(special_info, special_info)
            description = f"{category.replace('_', ' ')} displaying {special_info_readable}"
        elif category == "vehicle":
            vehicle_state, light_state, vehicle_base_type, vehicle_color = special_info
            light_state_description = vehicle_light_mapping.get(light_state, "unknown light state")

            if vehicle_state == "dynamic":
                if light_state == "RightBlinker":
                    intent_description = "preparing to turn right"
                elif light_state == "LeftBlinker":
                    intent_description = "preparing to turn left"
                elif light_state == "Brake":
                    intent_description = "slowing down or stopping"
                elif light_state == "Reverse":
                    intent_description = "reversing"
                else:
                    intent_description = "moving"
            else: 
                if light_state in ["Brake", "Reverse"]:
                    intent_description = "stopped"
                else:
                    intent_description = "stationary"

            if light_state_description in ["no significant lights are on", "unknown light state"]:
                description = f"{category.replace('_', ' ')} that is {intent_description}"
            else:
                description = f"{category.replace('_', ' ')} with {light_state_description}, indicating it is {intent_description}"
            if vehicle_color is not None:
                description = f"{vehicle_color} {description}"

        elif category == "walker":
            gender, age = special_info
            description = f"{category.replace('_', ' ')} who is a {age.lower()} {gender.lower()}"
        else:
            description = category.replace('_', ' ')

        result += f"The {seq_template[idx]} object is a {description}, located in the image at {bbox}, with a distance of {round(distance, 2)} meters from the ego vehicle.\n"
    
    if result == "":
        result = "There are no significant objects visible in this scene."
    return result

def generate_weather_template(weather_id):
    if weather_id in weather_templates:
        templates = weather_templates[weather_id]
        return random.choice(templates)
    else:
        return "The weather conditions are not available."

def generate_action_template(anno):
    steer = anno['steer']
    brake = anno['brake']
    
    # Steering templates
    steer_templates_right = [
        "Steer right to adjust your position on the road.",
        "Turn the wheel slightly to the right to stay on course.",
        "Gently steer to the right to correct your path.",
        "Guide the vehicle to the right to maintain your lane.",
        "Angle the steering wheel right to better align with the road.",
        "Adjust your steering to the right to navigate the curve.",
        "Shift the wheel to the right to keep the vehicle on track.",
        "Make a slight right turn to follow the road's contour.",
        "Direct the car to the right for a smoother drive.",
        "Correct your trajectory by steering right."
    ]
    
    steer_templates_left = [
        "Steer left to adjust your position on the road.",
        "Turn the wheel slightly to the left to stay on course.",
        "Gently steer to the left to correct your path.",
        "Guide the vehicle to the left to maintain your lane.",
        "Angle the steering wheel left to better align with the road.",
        "Adjust your steering to the left to navigate the curve.",
        "Shift the wheel to the left to keep the vehicle on track.",
        "Make a slight left turn to follow the road's contour.",
        "Direct the car to the left for a smoother drive.",
        "Correct your trajectory by steering left."
    ]
    
    steer_templates_straight = [
        "Maintain your current steering position.",
        "Keep the wheel steady and continue straight.",
        "Hold your current course, no steering adjustment needed.",
        "Steady as you go, keep the wheel centered.",
        "No need to adjust the steering, maintain your path.",
        "Continue straight ahead without altering the steering.",
        "Stay on course by holding the wheel steady.",
        "No steering change needed, keep going straight.",
        "Maintain your current direction without steering adjustments.",
        "Keep your hands steady on the wheel and drive straight."
    ]
    
    # Brake templates
    brake_templates_apply = [
        "Apply the brakes to slow down and maintain a safe speed.",
        "Press the brake pedal to reduce speed and ensure safety.",
        "Slow down by gently applying the brakes.",
        "Brake now to decrease your speed safely.",
        "Engage the brakes to avoid speeding.",
        "Reduce speed by pressing the brake pedal.",
        "Slow the vehicle by applying the brakes.",
        "Gently apply the brakes to reduce speed.",
        "Ease into the brakes to bring your speed down.",
        "Initiate braking to manage your speed."
    ]
    
    brake_templates_maintain = [
        "Maintain your current speed and be prepared to stop if needed.",
        "Keep your current speed but stay alert for any changes ahead.",
        "Continue at your current speed, but be ready to brake if necessary.",
        "Maintain speed, but keep an eye on the road for possible stops.",
        "Hold your speed, but stay prepared for any need to slow down.",
        "Keep your pace steady, but be vigilant for any stop signals.",
        "Continue at the current speed while staying alert.",
        "Maintain speed with caution, be ready to brake.",
        "Keep the current speed, but stay on the lookout for any obstacles.",
        "Continue at this speed, with readiness to brake if required."
    ]
    
    # Select random template based on steering and brake values
    if steer > 0.1:
        steer_template = random.choice(steer_templates_right)
    elif steer < -0.1:
        steer_template = random.choice(steer_templates_left)
    else:
        steer_template = random.choice(steer_templates_straight)
    
    if brake > 0.5:
        brake_template = random.choice(brake_templates_apply)
    else:
        brake_template = random.choice(brake_templates_maintain)
    
    return steer_template + " " + brake_template


def get_past_images(json_gz_path, num_images=20):
    
    path_parts = json_gz_path.split('/')
    anno = path_parts[-2]
    base_path = '/'.join(path_parts[:-2])
    current_image_number = int(path_parts[-1].split('.')[0])
    camera_folder = os.path.join(base_path, anno.replace('anno', 'camera/rgb_front'))
    
    image_paths = []
    for i in range(current_image_number - num_images + 1, current_image_number + 1, 4):
        if i > 0:
            image_filename = f"{i:05d}.jpg"
            image_path = os.path.join(camera_folder, image_filename)
            image_paths.append(image_path)
    return image_paths


def get_hash_key(video_path):
    key = video_path
    encoded_data = str(key).encode('utf-8')
    hash_object = hashlib.sha256(encoded_data)
    hash_hex = hash_object.hexdigest()
    return hash_hex