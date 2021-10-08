ROOT=../dataset
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=100
BATCH=8
LR=0.04

mkdir -p logs

identifier='08_10_2021_IRNet'

output_dir=results/seg_labels
output_imgs_list_path=datasets/data/train_aug.txt
irn_mask_root=~/data/irn_full_result/sem_seg
irn_imgs_name_path=datasets/data/train_aug.txt

#output_dir=results/seg_labels
#output_imgs_list_path=datasets/data/infer.txt
#irn_mask_root=/home/xuzhan/Documents/irn/data/cross_epoch_exp_5/sem_seg
#irn_imgs_name_path=datasets/data/infer.txt

# training with 2 GPUs
CUDA_VISLBLE_DEVICES=0,1 python main.py --data_root ${ROOT} \
                                        --model ${MODEL} \
                                        --gpu_id 0,1 \
                                        --output_dir ${output_dir} \
                                        --output_imgs_list_path ${output_imgs_list_path} \
                                        --irn_mask_root ${irn_mask_root} \
                                        --irn_imgs_name_path ${irn_imgs_name_path} \
                                        --amp \
                                        --total_itrs ${ITER} \
                                        --batch_size ${BATCH} \
                                        --lr ${LR}  \
                                        --crop_val |  tee logs/${identifier}.txt


## evalutation with crf
#CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val | tee logs/'eval'${identifier}.txt