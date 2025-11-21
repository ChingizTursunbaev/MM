# First, go to the directory where your script is
# cd /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/tools/

# # Set your INPUT paths (the data you just created)
# INPUT_TRAIN="/shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/interpolated_KP/interpolated.train"
# INPUT_DEV="/shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/interpolated_KP/interpolated.dev"
# INPUT_TEST="/shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/interpolated_KP/interpolated.test"

# # Set your OUTPUT paths (the new /tmp/ directory)
# # Set your OUTPUT paths (a permanent location in your project)
# OUTPUT_DIR="/shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/tokenized_KP_gpu"
# OUTPUT_TRAIN="$OUTPUT_DIR/tokenized.train"
# OUTPUT_DEV="$OUTPUT_DIR/tokenized.dev"
# OUTPUT_TEST="$OUTPUT_DIR/tokenized.test"

# # Create the new output directory
# mkdir -p $OUTPUT_DIR
# echo "Output directory created at $OUTPUT_DIR"

# # --- Run the scripts ---

# echo "Starting TOKENIZATION for TRAIN set... (This will be slow)"
# python tokenize_motion_dataset.py \
#   "$INPUT_TRAIN" \
#   "$OUTPUT_TRAIN" \
#   --kpoints 133

# echo "Starting TOKENIZATION for DEV set... (This will be slow)"
# python tokenize_motion_dataset.py \
#   "$INPUT_DEV" \
#   "$OUTPUT_DEV" \
#   --kpoints 133
  
# echo "Starting TOKENIZATION for TEST set... (This will be slow)"
# python tokenize_motion_dataset.py \
#   "$INPUT_TEST" \
#   "$OUTPUT_TEST" \
#   --kpoints 133

# echo "All tokenization is complete!"
# echo "Your new dataset is ready in $OUTPUT_DIR"

# # Make sure the output directory exists
# mkdir -p /tmp/xvoice_slr_data/interpolated_KP_10

# # 1. Process TRAINING set
# python interpolate_and_convert_to_motion_10.py \
#   /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/Phoenix-2014/phoenix-2014.train \
#   /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.train \
#   --axis_step=10 \
#   --kpoints=133

# # 2. Process DEV set
# python interpolate_and_convert_to_motion_10.py \
#   /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/Phoenix-2014/phoenix-2014.dev \
#   /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.dev \
#   --axis_step=10 \
#   --kpoints=133

# # 3. Process TEST set
# python interpolate_and_convert_to_motion_10.py \
#   /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/Phoenix-2014/phoenix-2014.test \
#   /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.test \
#   --axis_step=10 \
#   --kpoints=133



# Make sure the output directory exists
mkdir -p /tmp/xvoice_slr_data/tokenized_KP_10

# 1. Process TRAINING set
python tokenize_motion_dataset_10.py \
  /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.train \
  /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/tokenized_KP_gpu_10/phoenix-2014_10.train \
  --axis_step=10 \
  --kpoints=133

# 2. Process DEV set
python tokenize_motion_dataset_10.py \
  /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.dev \
  /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/tokenized_KP_gpu_10/phoenix-2014_10.dev \
  --axis_step=10 \
  --kpoints=133

# 3. Process TEST set
python tokenize_motion_dataset_10.py \
  /tmp/xvoice_slr_data/interpolated_KP_10/phoenix-2014_10.test \
  /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/tokenized_KP_gpu_10/phoenix-2014_10.test \
  --axis_step=10 \
  --kpoints=133

















# #!/bin/bash
# #
# # FINAL SCRIPT - USES /tmp AND RUNS SMALLEST SETS FIRST
# #

# # This command is CRITICAL. It stops the script if any command fails.
# set -e

# # === CONFIGURATION ===

# # 1. We will use /tmp for scratch, as we have write permission and 238GB free.
# #    We create a personal folder inside /tmp for safety.
# SCRATCH_BASE_PATH="/tmp/xvoice_slr_data" # <-- THIS IS THE CORRECT PATH

# # 2. Your input directory (this is fine)
# INPUT_DIR="/shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/Phoenix-2014"

# # 3. Set the (new) output directory to be on the /tmp scratch space
# OUTPUT_DIR="$SCRATCH_BASE_PATH/interpolated_KP"

# # 4. Set the (new) temporary directory for the python script
# export TMPDIR="$SCRATCH_BASE_PATH/temp_job_files"
# # =======================

# # Clean up any old runs (optional, but good practice)
# rm -rf $SCRATCH_BASE_PATH

# # Create the output and temp directories on the /tmp drive
# mkdir -p $OUTPUT_DIR
# mkdir -p $TMPDIR
# echo "Created temporary directories in $SCRATCH_BASE_PATH"

# echo "=============================================="
# echo "Starting interpolation for TEST set (Smallest)"
# echo "Temp directory: $TMPDIR"
# echo "Outputting to: $OUTPUT_DIR/interpolated.test"
# echo "=============================================="
# python interpolate_and_convert_to_motion.py \
#   "$INPUT_DIR/phoenix-2014.test" \
#   "$OUTPUT_DIR/interpolated.test" \
#   --kpoints 133 \
#   --axis_step 1.0

# echo "=============================================="
# echo "TEST SET SUCCEEDED. Starting DEV set..."
# echo "Temp directory: $TMPDIR"
# echo "Outputting to: $OUTPUT_DIR/interpolated.dev"
# echo "=============================================="
# python interpolate_and_convert_to_motion.py \
#   "$INPUT_DIR/phoenix-2014.dev" \
#   "$OUTPUT_DIR/interpolated.dev" \
#   --kpoints 133 \
#   --axis_step 1.0

# echo "=============================================="
# echo "DEV SET SUCCEEDED. Starting TRAIN set (Largest)"
# echo "Temp directory: $TMPDIR"
# echo "Outputting to: $OUTPUT_DIR/interpolated.train"
# echo "=============================================="
# python interpolate_and_convert_to_motion.py \
#   "$INPUT_DIR/phoenix-2014.train" \
#   "$OUTPUT_DIR/interpolated.train" \
#   --kpoints 133 \
#   --axis_step 1.0

# echo "=============================================="
# echo "All interpolation complete."
# echo "Your final files are in: $OUTPUT_DIR"
# echo ""
# echo "!!! IMPORTANT !!!"
# echo "Files in /tmp can be deleted on reboot."
# echo "Copy your data to your home directory with:"
# echo "cp -r $OUTPUT_DIR /shared/home/xvoice/Chingiz/slr_project_ms_corr_mamba/data/"
# echo "=============================================="