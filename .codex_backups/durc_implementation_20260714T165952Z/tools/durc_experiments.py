"""Canonical DURC experiment identifiers and model configuration files."""

MODEL_FILES = {
    "A1": "yolov13-durc-a1-drsc-p2.yaml",
    "A2": "yolov13-durc-a2-drsc-p3.yaml",
    "A3": "yolov13-durc-a3-drsc-p2p3.yaml",
    "B1": "yolov13-durc-b1-hrct-p3.yaml",
    "B2": "yolov13-durc-b2-hrct-p4.yaml",
    "B3": "yolov13-durc-b3-hrct-p3p4.yaml",
    "B4": "yolov13-durc-b4-hrct-all.yaml",
    "C1": "yolov13-durc-c1-nudfl.yaml",
    "C2": "yolov13-durc-c2-softlabel.yaml",
    "C3": "yolov13-durc-c3-uncertainty.yaml",
    "C4": "yolov13-durc-c4-full.yaml",
}

STAGE_ORDER = tuple(MODEL_FILES)

MAIN_FILES = {
    "S1": "yolov13-durc-s1.yaml",
    "S2": "yolov13-durc-s2.yaml",
    "S3": "yolov13-durc-s3.yaml",
}
