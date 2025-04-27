echo "Running UpGIN baseline experiments..."

# AID1798
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_deepgin_scaffaug_st.py --dataset AID1798 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID463087
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_deepgin_scaffaug_st.py --dataset AID463087 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID488997
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_deepgin_scaffaug_st.py --dataset AID488997 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID2689
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_deepgin_scaffaug_st.py --dataset AID2689 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID485290
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_deepgin_scaffaug_st.py --dataset AID485290 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# GAT experiments
# AID1798
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_gat_scaffaug_st.py --dataset AID1798 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID463087
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_gat_scaffaug_st.py --dataset AID463087 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID488997
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_gat_scaffaug_st.py --dataset AID488997 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID2689
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_gat_scaffaug_st.py --dataset AID2689 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

# AID485290
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python finetune_gat_scaffaug_st.py --dataset AID485290 --split "$split" --sampling_method uniform_sabs &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 5 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait
