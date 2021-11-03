ROOT=../dataset
#MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
MODEL=deeplabv3plus_mobilenet
ITER=300
BATCH=12
LR=0.06

mkdir -p logs


soft_weight_root=/home/xuzhan/Documents/IRNet_Lucas/data/soft_weights_full
seg_labels_root=/home/xuzhan/Downloads/IRNet_seg5_full/IRNet_server_exp1_CAM_epoch5/sem_seg
train_file_path=./datasets/data/infer.txt
val_file_path=./datasets/data/infer.txt
output_imgs_list_path=./datasets/data/infer.txt

num_workers=8

for exp_no in 1
do
  identifier='test_save_seg_exp'${exp_no}
  output_dir=/home/xuzhan/Desktop/${identifier}
  mkdir -p ${output_dir}
  # training with 2 GPUs
  CUDA_VISLBLE_DEVICES=0,1 python main.py --data_root ${ROOT} \
                                          --model ${MODEL} \
                                          --gpu_id 0,1 \
                                          --output_dir ${output_dir} \
                                          --output_imgs_list_path ${output_imgs_list_path} \
                                          --seg_labels_root ${seg_labels_root} \
                                          --train_file_path ${train_file_path} \
                                          --val_file_path ${val_file_path} \
                                          --amp \
                                          --total_itrs ${ITER} \
                                          --batch_size ${BATCH} \
                                          --lr ${LR}  \
                                          --num_workers ${num_workers} \
                                          --soft_weight_root ${soft_weight_root} \
                                          --crop_val 2>&1 | tee output_dir/${identifier}_log.txt

done



## evalutation with crf
#CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val | tee logs/'eval'${identifier}.txt