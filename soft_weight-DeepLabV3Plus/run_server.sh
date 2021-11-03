ROOT=../dataset
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=5000
BATCH=20
LR=0.05

mkdir -p logs


soft_weight_root=/home/xzhan/data/full_soft_weights
seg_labels_root=~/data/irn_full_result/sem_seg
train_file_path=./datasets/data/train_aug.txt
val_file_path=./datasets/data/infer.txt
output_imgs_list_path=./datasets/data/train_aug.txt

num_workers=8

for exp_no in 1 2 3 4 5
do
  identifier='IRNet_full_2_DLV3+_sw_exp'${exp_no}
  output_dir='/home/xzhan/data/IRNet_full_seg_2_DLV3+_sw_exp'${exp_no}
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
                                          --crop_val 2>&1 | tee logs/${identifier}_log.txt

done







## evalutation with crf
#CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val | tee logs/'eval'${identifier}.txt