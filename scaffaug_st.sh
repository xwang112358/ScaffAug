# # echo "Start finetuning GCN with pseudo labels"
# # AID1798
# echo "AID1798"
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv1 & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv2 & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv3 & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv4 & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv5 
# wait  
  
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split random_cv5 & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split scaffold_seed1  & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split scaffold_seed2  & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split scaffold_seed3  & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split scaffold_seed4  & \
# python finetune_gcn_pseudo_label.py --dataset AID1798 --split scaffold_seed5  
# wait

# # AID463087
# echo "AID463087"
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split random_cv1 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split random_cv2 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split random_cv3 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split random_cv4 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split random_cv5
# wait

# python finetune_gcn_pseudo_label.py --dataset AID463087 --split scaffold_seed1 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split scaffold_seed2 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split scaffold_seed3 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split scaffold_seed4 & \
# python finetune_gcn_pseudo_label.py --dataset AID463087 --split scaffold_seed5 
# wait

# # AID488997
# echo "AID488997"
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split random_cv1 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split random_cv2 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split random_cv3 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split random_cv4 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split random_cv5 
# wait

# python finetune_gcn_pseudo_label.py --dataset AID488997 --split scaffold_seed1 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split scaffold_seed2 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split scaffold_seed3 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split scaffold_seed4 & \
# python finetune_gcn_pseudo_label.py --dataset AID488997 --split scaffold_seed5 
# wait

# # AID2689
# echo "AID2689"
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split random_cv1 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split random_cv2 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split random_cv3 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split random_cv4 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split random_cv5 
# wait

# python finetune_gcn_pseudo_label.py --dataset AID2689 --split scaffold_seed1 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split scaffold_seed2 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split scaffold_seed3 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split scaffold_seed4 & \
# python finetune_gcn_pseudo_label.py --dataset AID2689 --split scaffold_seed5 
# wait

# # AID485290
# echo "AID485290"
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split random_cv1 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split random_cv2 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split random_cv3 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split random_cv4 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split random_cv5 
# wait

# python finetune_gcn_pseudo_label.py --dataset AID485290 --split scaffold_seed1 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split scaffold_seed2 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split scaffold_seed3 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split scaffold_seed4 & \
# python finetune_gcn_pseudo_label.py --dataset AID485290 --split scaffold_seed5 
# wait

# GIN finetuning with pseudo labels
echo "Start finetuning GIN with pseudo labels"

# # AID1798
# echo "AID1798 GIN"
# python finetune_gin_pseudo_label.py --dataset AID1798 --split random_cv1 & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split random_cv2 & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split random_cv3 & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split random_cv4 & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split random_cv5 
# wait  
  
# python finetune_gin_pseudo_label.py --dataset AID1798 --split scaffold_seed1  & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split scaffold_seed2  & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split scaffold_seed3  & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split scaffold_seed4  & \
# python finetune_gin_pseudo_label.py --dataset AID1798 --split scaffold_seed5  
# wait

# AID463087
echo "AID463087 GIN"
python finetune_gin_pseudo_label.py --dataset AID463087 --split random_cv1 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split random_cv2 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split random_cv3 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split random_cv4 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split random_cv5
wait

python finetune_gin_pseudo_label.py --dataset AID463087 --split scaffold_seed1 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split scaffold_seed2 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split scaffold_seed3 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split scaffold_seed4 & \
python finetune_gin_pseudo_label.py --dataset AID463087 --split scaffold_seed5 
wait

# AID488997
echo "AID488997 GIN"
python finetune_gin_pseudo_label.py --dataset AID488997 --split random_cv1 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split random_cv2 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split random_cv3 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split random_cv4 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split random_cv5 
wait

python finetune_gin_pseudo_label.py --dataset AID488997 --split scaffold_seed1 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split scaffold_seed2 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split scaffold_seed3 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split scaffold_seed4 & \
python finetune_gin_pseudo_label.py --dataset AID488997 --split scaffold_seed5 
wait

# AID2689
echo "AID2689 GIN"
python finetune_gin_pseudo_label.py --dataset AID2689 --split random_cv1 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split random_cv2 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split random_cv3 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split random_cv4 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split random_cv5 
wait

python finetune_gin_pseudo_label.py --dataset AID2689 --split scaffold_seed1 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split scaffold_seed2 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split scaffold_seed3 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split scaffold_seed4 & \
python finetune_gin_pseudo_label.py --dataset AID2689 --split scaffold_seed5 
wait

# AID485290
echo "AID485290 GIN"
python finetune_gin_pseudo_label.py --dataset AID485290 --split random_cv1 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split random_cv2 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split random_cv3 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split random_cv4 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split random_cv5 
wait

python finetune_gin_pseudo_label.py --dataset AID485290 --split scaffold_seed1 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split scaffold_seed2 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split scaffold_seed3 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split scaffold_seed4 & \
python finetune_gin_pseudo_label.py --dataset AID485290 --split scaffold_seed5 
wait


