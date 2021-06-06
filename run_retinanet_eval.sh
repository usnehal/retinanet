set -e
set -x

print_red () {
  echo -e "\e[1;31m$1\e[0m"
}

print_green () {
  echo -e "\e[1;34m$1\e[0m"
}

case $HOSTNAME in
    (LTsuphale-NC2JM) 
        print_red "We are in WSL"
        export STORAGE_BUCKET=/home/suphale/snehal_bucket
        export PYTHONPATH="/home/suphale/retinanet/models:/home/suphale/retinanet/tpu/models:${PYTHONPATH}"
	    USE_TPU=off
        ;;
    (snehal-vm-tpu) 
        print_red "We are in vm-tpu"
        print_green "Install dependencies"
        git config --global user.email "snehal.v.uphale@gmail.com"
        git config --global user.name "Snehal Uphale"
        sudo apt-get install -y python3-tk
        pip3 install --user Cython matplotlib opencv-python-headless pyyaml Pillow
        pip3 install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
        pip3 install --user -U gast
        pip3 install --user -U absl-py
        #sudo pip3 install --user -r /usr/share/models/official/requirements.txt
        export STORAGE_BUCKET=gs://snehal_bucket
        export PYTHONPATH="/home/suphale/retinanet/models:/home/suphale/retinanet/tpu/models:${PYTHONPATH}"
	    USE_TPU=on
        ;;
    (*)   
        print_red "We are somewhere";;
esac

export DATA_DIR=${STORAGE_BUCKET}/coco
export MODEL_DIR=${STORAGE_BUCKET}/retinanet-train
export RESNET_CHECKPOINT=${STORAGE_BUCKET}/resnet50-checkpoint-2018-02-07
export TRAIN_FILE_PATTERN=${DATA_DIR}/train-*
export EVAL_FILE_PATTERN=${DATA_DIR}/val-*
export VAL_JSON_FILE=${DATA_DIR}/instances_val2017.json
#export PYTHONPATH=/usr/share/models
print_green "Start Training brief"
if [ "$USE_TPU" = on ]; then
    print_green "Start brief Training"
    python3 ~/retinanet/models/official/vision/detection/main.py \
        --strategy_type=tpu --tpu=${TPU_NAME} \
        --model_dir=${MODEL_DIR} \
        --mode="train" \
        --params_override="{ type: retinanet, train: { total_steps: 10, checkpoint: { path: ${RESNET_CHECKPOINT}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN} }, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: 5000 } }"
    print_green "Start Training"
    export EVAL_SAMPLES=5000
    python3 ~/retinanet/models/official/vision/detection/main.py \
        --strategy_type=tpu --tpu=${TPU_NAME} \
        --model_dir=${MODEL_DIR} \
        --checkpoint_path=${MODEL_DIR} \
        --mode=eval_once \
        --params_override="{ type: retinanet, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: ${EVAL_SAMPLES} } }"
else
    print_green "Skipping training"
    # print_green "Start brief Training"
    # python3 ~/retinanet/models/official/vision/detection/main.py \
    #     --model_dir=${MODEL_DIR} \
    #     --mode="train" \
    #     --params_override="{ type: retinanet, train: { total_steps: 10, checkpoint: { path: ${RESNET_CHECKPOINT}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN} }, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: 5000 } }"
    # print_green "Start Training"
    # export EVAL_SAMPLES=5000
    # python3 ~/retinanet/models/official/vision/detection/main.py \
    #     --model_dir=${MODEL_DIR} \
    #     --checkpoint_path=${MODEL_DIR} \
    #     --mode=eval_once \
    #     --params_override="{ type: retinanet, eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: ${EVAL_SAMPLES} } }"
fi


#print_green "Save model"
#EXPORT_DIR=${STORAGE_BUCKET}/saved_model
#CHECKPOINT_PATH=${MODEL_DIR}
#PARAMS_OVERRIDE=""
#BATCH_SIZE=1
#INPUT_TYPE="image_bytes"
#INPUT_NAME="input"
#INPUT_IMAGE_SIZE="640,640"

#python3 ~/retinanet/tpu/models/official/detection/export_saved_model.py \
#  --export_dir="${EXPORT_DIR?}" \
#  --checkpoint_path="${CHECKPOINT_PATH?}" \
#  --params_override="${PARAMS_OVERRIDE?}" \
#  --batch_size=${BATCH_SIZE?} \
#  --input_type="${INPUT_TYPE?}" \
#  --input_name="${INPUT_NAME?}" \
#  --input_image_size="${INPUT_IMAGE_SIZE?}" \

print_green "Inference"
MODEL="retinanet"
IMAGE_SIZE=640
CHECKPOINT_PATH="${MODEL_DIR}/ctl_step_10.ckpt-2"
PARAMS_OVERRIDE=""  # if any.
LABEL_MAP_FILE="/home/suphale/retinanet/tpu/models/official/detection/datasets/coco_label_map.csv"
IMAGE_FILE_PATTERN=/home/suphale/retinanet/000000111117.jpg
OUTPUT_HTML="./test.html"
python3 ~/retinanet/tpu/models/official/detection/inference.py \
  --model="${MODEL?}" \
  --image_size=${IMAGE_SIZE?} \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --label_map_file="${LABEL_MAP_FILE?}" \
  --image_file_pattern="${IMAGE_FILE_PATTERN?}" \
  --output_html="${OUTPUT_HTML?}" \
  --max_boxes_to_draw=10 \
  --min_score_threshold=0.05


