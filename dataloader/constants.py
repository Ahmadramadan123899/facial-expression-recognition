import os

CK_EMOTIONS_LIST = ["Neutral", "Anger", "Contempt", "Disgust", "Fear",
                    "Happiness", "Sadness", "Surprise"]

# 2D-FAN 68-landmark layout (dlib/IBUG order):
#   0-16  : jaw line
#   17-21 : left eyebrow
#   22-26 : right eyebrow
#   27-35 : nose bridge + tip
#   36-41 : left eye
#   42-47 : right eye
#   48-67 : mouth (outer + inner)

LIST_INDICES_LIPS_68 = list(range(48, 68))          # 20 points — mouth

LIST_INDICES_LEFT_EYE_68  = list(range(36, 42))     # 6 points
LIST_INDICES_RIGHT_EYE_68 = list(range(42, 48))     # 6 points
LIST_INDICES_ALL_EYES_68  = list(range(36, 48))     # 12 points

LIST_INDICES_LEFT_BROW_68  = list(range(17, 22))    # 5 points
LIST_INDICES_RIGHT_BROW_68 = list(range(22, 27))    # 5 points

LIST_INDICES_NOSE_68   = list(range(27, 36))        # 9 points
LIST_INDICES_JAW_68    = list(range(0, 17))         # 17 points — jaw

# Subsets used as graph_type equivalents

# g_bouche equivalent (lips only)
LIST_FAN_LIPS = LIST_INDICES_LIPS_68                # 20 pts

# g_yeux_bouche equivalent (eyes + brows + mouth)
LIST_FAN_EYES_MOUTH = (
    LIST_INDICES_LEFT_BROW_68 +
    LIST_INDICES_RIGHT_BROW_68 +
    LIST_INDICES_ALL_EYES_68 +
    LIST_INDICES_LIPS_68
)  # 42 pts

# g_topologique equivalent (eyes + brows + nose + mouth, no jaw)
LIST_FAN_TOPOLOGIQUE = (
    LIST_INDICES_LEFT_BROW_68 +
    LIST_INDICES_RIGHT_BROW_68 +
    LIST_INDICES_ALL_EYES_68 +
    LIST_INDICES_NOSE_68 +
    LIST_INDICES_LIPS_68
)  # 51 pts — same as the paper's 49 (we keep 51, paper drops 2 iris pts)

# g_complet equivalent (all 68)
LIST_FAN_COMPLET = list(range(68))                  # 68 pts


# Old MediaPipe constants, kept for the old gnn.py / build_dataset.py pipeline

LIST_CK_SUBSET_51 = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                     65, 66, 67]

LIST_MP_SUBSET_146 = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249,
    251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473,
    474, 475, 476, 477
]

LIST_MP_SUBSET_136 = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249,
    251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466
]

LIST_INDICES_LIPS        = [0, 267, 269, 270, 14, 13, 17, 146, 402, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311,
                            312, 185, 314, 61, 317, 318, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
LIST_INDICES_LEFT_EYE    = [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
LIST_INDICES_LEFT_BROW   = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
LIST_INDICES_LEFT_IRIS   = [474, 475, 476, 477]
LIST_INDICES_RIGHT_EYE   = [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
LIST_INDICES_RIGHT_BROW  = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]
LIST_INDICES_RIGHT_IRIS  = [472, 469, 470, 471]
LIST_INDICES_ALL_EYES    = [0, 263, 7, 267, 269, 270, 14, 13, 17, 276, 282, 283, 285, 33, 291, 37, 293, 39, 40, 295, 296,
                            300, 46, 308, 52, 310, 311, 312, 53, 314, 55, 61, 317, 318, 63, 321, 65, 66, 324, 70, 78, 334,
                            80, 81, 82, 336, 84, 87, 88, 91, 95, 105, 362, 107, 373, 374, 375, 380, 381, 382, 384, 385,
                            386, 387, 388, 133, 390, 398, 144, 145, 146, 402, 405, 409, 153, 154, 155, 157, 158, 415,
                            160, 161, 159, 163, 173, 178, 181, 185, 191, 466, 469, 470, 471, 472, 474, 475, 476, 477,
                            246, 249]
LIST_MP_SUBSET_100 = [0, 7, 13, 14, 17, 33, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 87, 88,
                      91, 95, 105, 107, 133, 144, 145, 146, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 178, 181,
                      185, 191, 246, 249, 263, 267, 269, 270, 276, 282, 283, 285, 291, 293, 295, 296, 300, 308, 310, 311,
                      312, 314, 317, 318, 321, 324, 334, 336, 362, 373, 374, 375, 380, 381, 382, 384, 385, 386, 387, 388,
                      390, 398, 402, 405, 409, 415, 466, 8, 168, 6, 197, 195, 5, 4, 1]

DICT_LISTS_INDICES = {
    # 2D-FAN subsets, used by paper pipeline
    "FAN_COMPLET"    : LIST_FAN_COMPLET,       # 68 pts — g_complet equivalent
    "FAN_TOPO"       : LIST_FAN_TOPOLOGIQUE,   # 51 pts — g_topologique equivalent
    "FAN_EYES_MOUTH" : LIST_FAN_EYES_MOUTH,    # 42 pts — g_yeux_bouche equivalent
    "FAN_LIPS"       : LIST_FAN_LIPS,          # 20 pts — g_bouche equivalent
    # MediaPipe subsets, old pipeline, kept for backward compatibility
    "LIPS"      : LIST_INDICES_LIPS,
    "LEFT_EYE"  : LIST_INDICES_LEFT_EYE,
    "LEFT_BROW" : LIST_INDICES_LEFT_BROW,
    "LEFT_IRIS" : LIST_INDICES_LEFT_IRIS,
    "RIGHT_EYE" : LIST_INDICES_RIGHT_EYE,
    "RIGHT_BROW": LIST_INDICES_RIGHT_BROW,
    "RIGHT_IRIS": LIST_INDICES_RIGHT_IRIS,
    "ALL_EYES"  : LIST_INDICES_ALL_EYES,
    "MP_SUBSET_146": LIST_MP_SUBSET_146,
    "MP_SUBSET_136": LIST_MP_SUBSET_136,
    "CK_SUBSET_51" : LIST_CK_SUBSET_51,
}

# 2D-FAN graph edges (fully connected within each region) are generated at
# runtime from these subsets by from_nodes_create_list_edges_fan() in
# build_dataset_paper.py.
