import torch

# Data related attributes
NUM_TO_ATTR: dict[int, str] = {
    0: "5_o_Clock_Shadow",
    1: "Arched_Eyebrows",
    2: "Attractive",
    3: "Bags_Under_Eyes",
    4: "Bald",
    5: "Bangs",
    6: "Big_Lips",
    7: "Big_Nose",
    8: "Black_Hair",
    9: "Blond_Hair",
    10: "Blurry",
    11: "Brown_Hair",
    12: "Bushy_Eyebrows",
    13: "Chubby",
    14: "Double_Chin",
    15: "Eyeglasses",
    16: "Goatee",
    17: "Gray_Hair",
    18: "Heavy_Makeup",
    19: "High_Cheekbones",
    20: "Male",
    21: "Mouth_Slightly_Open",
    22: "Mustache",
    23: "Narrow_Eyes",
    24: "No_Beard",
    25: "Oval_Face",
    26: "Pale_Skin",
    27: "Pointy_Nose",
    28: "Receding_Hairline",
    29: "Rosy_Cheeks",
    30: "Sideburns",
    31: "Smiling",
    32: "Straight_Hair",
    33: "Wavy_Hair",
    34: "Wearing_Earrings",
    35: "Wearing_Hat",
    36: "Wearing_Lipstick",
    37: "Wearing_Necklace",
    38: "Wearing_Necktie",
    39: "Young",
}


ATTR_TO_NUM = {
    "5_o_Clock_Shadow": 0,
    "Arched_Eyebrows": 1,
    "Attractive": 2,
    "Bags_Under_Eyes": 3,
    "Bald": 4,
    "Bangs": 5,
    "Big_Lips": 6,
    "Big_Nose": 7,
    "Black_Hair": 8,
    "Blond_Hair": 9,
    "Blurry": 10,
    "Brown_Hair": 11,
    "Bushy_Eyebrows": 12,
    "Chubby": 13,
    "Double_Chin": 14,
    "Eyeglasses": 15,
    "Goatee": 16,
    "Gray_Hair": 17,
    "Heavy_Makeup": 18,
    "High_Cheekbones": 19,
    "Male": 20,
    "Mouth_Slightly_Open": 21,
    "Mustache": 22,
    "Narrow_Eyes": 23,
    "No_Beard": 24,
    "Oval_Face": 25,
    "Pale_Skin": 26,
    "Pointy_Nose": 27,
    "Receding_Hairline": 28,
    "Rosy_Cheeks": 29,
    "Sideburns": 30,
    "Smiling": 31,
    "Straight_Hair": 32,
    "Wavy_Hair": 33,
    "Wearing_Earrings": 34,
    "Wearing_Hat": 35,
    "Wearing_Lipstick": 36,
    "Wearing_Necklace": 37,
    "Wearing_Necktie": 38,
    "Young": 39,
}

ATTR_TO_SPANISH = {
    "Arched_Eyebrows": "Cejas arqueadas",
    "Attractive": "Atractivo/a",
    "Bald": "Calvo/a",
    "Bangs": "Flequillo",
    "Big_Lips": "Labios gruesos",
    "Big_Nose": "Nariz grande",
    "Blond_Hair": "Pelo rubio",
    "Bushy_Eyebrows": "Cejas pobladas",
    "Chubby": "Regordete/a",
    "Eyeglasses": "Gafas",
    "Gray_Hair": "Pelo gris",
    "High_Cheekbones": "Pómulos altos",
    "Male": "Masculino",
    "Narrow_Eyes": "Ojos estrechos",
    "No_Beard": "Sin barba",
    "Pale_Skin": "Piel pálida",
    "Smiling": "Sonriendo",
    "Wavy_Hair": "Pelo ondulado",
    "Wearing_Lipstick": "Pintalabios",
    "Young": "Joven",
}

# Dataset for eval in NATTRS
NATTRS_NUM_TO_IDX = {
    "Blond_Hair": 0,
    "Bangs": 1,
    "Chubby": 2,
    "Smiling": 3,
    "Young": 4,
    "Male": 5,
    "Heavy_Makeup": 6,
    "No_Beard": 7,
    "Wearing_Lipstick": 8,
    "Big_Nose": 9,
    "High_Cheekbones": 10,
    "Pale_Skin":11,
    "Eyeglasses": 12,
    "Arched_Eyebrows": 13,
    "Narrow_Eyes": 14,
}

# Paths
LATENT_PATH = "./prepared_data/celebahq_dlatents_psp.npy"
LABEL_FILE = "./prepared_data/celebahq_anno.npy"
LOG_DIR = ".models/logs/"
STYLEGAN = ".models/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
CLASSIFIER = "./models/latent_classifier_epoch_20.pth"

# Torch config
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")