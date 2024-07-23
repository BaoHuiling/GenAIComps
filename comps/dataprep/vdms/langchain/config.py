# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re

# Vdms Collection Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "video-test")
# Vdms Connection Information
VDMS_URL = os.getenv("VDMS_URL", "http://0.0.0.0:55555")

def extract_vdms_conn_from_env():
    vdms_host = os.getenv("VDMS_HOST", None)
    vdms_port = os.getenv("VDMS_PORT", None)
    if vdms_host and vdms_port:
        return vdms_host, int(vdms_port)

    match = re.match(r'http://(.*):(\d+)', VDMS_URL)
    if match:
        ip = match.group(1)
        port = match.group(2)
        return ip, int(port)

    raise ValueError("Invalid VDMS_URL")

VDMS_HOST, VDMS_PORT = extract_vdms_conn_from_env()     

# Read MeanCLIP
meanclip_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding/meanclip_config/clip_meanAgg.json")
meanclip_cfg_json = json.load(open(meanclip_cfg_path, 'r'))
MEANCLIP_CFG = argparse.Namespace(**meanclip_cfg_json)
