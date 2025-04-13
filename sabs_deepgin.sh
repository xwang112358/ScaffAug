echo "Running UpGIN baseline experiments..."

# AID1798
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split random_cv5 --sampling_method sabs
wait

python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split scaffold_seed5 --sampling_method sabs
wait

# AID463087
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split random_cv1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split random_cv2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split random_cv3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split random_cv4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split random_cv5 --sampling_method sabs
wait

python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split scaffold_seed1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split scaffold_seed2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split scaffold_seed3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split scaffold_seed4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split scaffold_seed5 --sampling_method sabs
wait

# AID488997
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split random_cv1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split random_cv2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split random_cv3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split random_cv4 --sampling_method sabs & \   
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split random_cv5 --sampling_method sabs
wait

python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split scaffold_seed1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split scaffold_seed2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split scaffold_seed3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split scaffold_seed4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split scaffold_seed5 --sampling_method sabs
wait

# AID2689
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split random_cv1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split random_cv2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split random_cv3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split random_cv4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split random_cv5 --sampling_method sabs
wait

python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split scaffold_seed1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split scaffold_seed2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split scaffold_seed3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split scaffold_seed4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split scaffold_seed5 --sampling_method sabs
wait

# AID485290
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split random_cv1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split random_cv2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split random_cv3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split random_cv4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split random_cv5 --sampling_method sabs
wait

python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed1 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed2 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed3 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed4 --sampling_method sabs & \
python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split scaffold_seed5 --sampling_method sabs
wait