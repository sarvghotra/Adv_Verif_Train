MODEL=$1
FOLDER_GSM8K=$2
OUTPUT_FOLDER=$3

mkdir -p $OUTPUT_FOLDER
python reward_model_inference.py \
    --model_name_or_path $MODEL\
    --data_folder $FOLDER_GSM8K \
    --output_folder $OUTPUT_FOLDER
    

# FOLDER_MATH=$3
# mkdir -p $FOLDER_MATH
# python reward_model_inference_MATH.py \
#     --model_name_or_path $MODEL\
#     --folder_new $FOLDER_MATH