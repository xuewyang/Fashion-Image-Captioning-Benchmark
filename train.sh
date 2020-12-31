#DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/COCO
#CUDA_VISIBLE_DEVICES='0' python tools/train.py --id fc --caption_model newfc --input_json $DATA_FOLDER/cocotalk.json \
#--input_fc_dir $DATA_FOLDER/cocobu/cocobu_fc --input_att_dir $DATA_FOLDER/cocobu/cocobu_att \
#--input_label_h5 $DATA_FOLDER/cocotalk_label.h5 --batch_size 500 --learning_rate 5e-4 \
#--learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc2 --save_checkpoint_every 6000 \
#--val_images_use 5000 --max_epochs 30

#DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/FACAD/images/
#OUTPUT_FOLDER=/home/xuewyang/Xuewen/Research/model/fashion/captioning/newfc
#CUDA_VISIBLE_DEVICES='1,0' python tools/train.py --id fc --caption_model newfc --batch_size 120 --learning_rate 5e-4 \
#--learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $OUTPUT_FOLDER --save_checkpoint_every 2000 \
#--val_images_use 5000 --max_epochs 30 --data_folder $DATA_FOLDER --max_length 30 --language_eval 1

CUDA_VISIBLE_DEVICES='1' python tools/train.py --cfg configs/updown/updown.yml --id updown
