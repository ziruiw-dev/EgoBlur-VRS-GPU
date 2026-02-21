#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
import shlex

def run_egoblur():
    parser = argparse.ArgumentParser(
        description="EgoBlur VRS GPU Processing Wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments (using named flags for clarity)
    parser.add_argument("--input-dir", type=str, required=True, help="Host directory containing input VRS files")
    parser.add_argument("--output-dir", type=str, required=True, help="Host directory for blurred VRS files")
    parser.add_argument("--vrs-name", type=str, required=True, help="Name of the VRS file to process (relative to input_dir)")
    
    # Optional arguments for volumes
    default_ckpt_dir = "/mnt/storage0/projects/oxford_day_and_night/ckpts/EgoBlur"
    parser.add_argument("--ckpt-dir", type=str, default=default_ckpt_dir,
                        help="Host directory containing model checkpoints (face and lp .jit files)")
    
    # Model thresholds
    parser.add_argument("--face-conf", type=float, default=0.8, help="Face model confidence threshold")
    parser.add_argument("--lp-conf", type=float, default=0.8, help="License plate model confidence threshold")
    
    args = parser.parse_args()

    # Resolve absolute paths for volumes on the host
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    ckpt_path = Path(args.ckpt_dir).resolve()

    # Validation
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        sys.exit(1)
    
    # Ensure output directory exists on host
    output_path.mkdir(parents=True, exist_ok=True)

    # Internal container paths
    container_raw_dir = "/vrs/raw"
    container_blur_dir = "/vrs/blur"
    container_ckpt_dir = "/workspace/ckpts"
    binary_path = "/workspace/repos/EgoBlur-VRS-GPU/gen1/tools/vrs_mutation/build/ego_blur_vrs_mutation"

    # Construct the internal command to run inside the container
    # Using shlex.quote for safety with paths/filenames
    vrs_name_quoted = shlex.quote(args.vrs_name)
    
    inner_cmd = (
        f"{binary_path} "
        f"-i {container_raw_dir}/{vrs_name_quoted} "
        f"-o {container_blur_dir}/{vrs_name_quoted} "
        f"-f {container_ckpt_dir}/ego_blur_face.jit "
        f"-l {container_ckpt_dir}/ego_blur_lp.jit "
        f"--use-gpu "
        f"--face-model-confidence-threshold {args.face_conf} "
        f"--license-plate-model-confidence-threshold {args.lp_conf}"
    )

    # Docker Compose command
    docker_cmd = [
        "docker", "compose", "run", "--rm",
        "-v", f"{input_path}:{container_raw_dir}",
        "-v", f"{output_path}:{container_blur_dir}",
        "-v", f"{ckpt_path}:{container_ckpt_dir}",
        "egoblur-vrs-gpu",
        "bash", "-c", inner_cmd
    ]

    print(f"\n{'='*60}")
    print(f"EgoBlur Processing")
    print(f"{'='*60}")
    print(f"Input File:  {input_path}/{args.vrs_name}")
    print(f"Output File: {output_path}/{args.vrs_name}")
    print(f"CKPT Dir:    {ckpt_path}")
    print(f"Face Conf:   {args.face_conf}")
    print(f"LP Conf:     {args.lp_conf}")
    print(f"Using GPU:   True")
    print(f"{'='*60}\n")

    try:
        # Run the command and pipe output to terminal
        subprocess.run(docker_cmd, check=True)
        print(f"\nProcessing complete: {output_path}/{args.vrs_name}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: EgoBlur failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    run_egoblur()
