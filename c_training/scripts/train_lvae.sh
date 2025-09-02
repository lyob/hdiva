#!/bin/bash -l

#SBATCH --job-name=lvae
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -C a100-80gb
# This should always be 1!
#SBATCH --ntasks-per-node=1
# This is the physical number of GPUs per node
#SBATCH --gpus-per-node=1
# You can vary that one if you see you need more or less CPU cores per gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --time=1:00:00

jobid=$SLURM_JOB_ID
current_dir=$PWD
main_dir="/mnt/home/blyo1/ceph/projects/hdiva"
outdir="$main_dir/c_training/cluster_logs/$jobid"
mkdir -p $outdir
cd $main_dir

module purge
module load modules/2.3
module load python
module load cuda cudnn nccl
source ~/venvs/py310/bin/activate  # modules/2.3 and python/3.10.13

master_node=$SLURMD_NODENAME

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno1

echo "Starting training"
srun --error="$main_dir/c_training/cluster_logs/${jobid}/err.err" \
	 --output="$main_dir/c_training/cluster_logs/${jobid}/out.out" \
		python `which torchrun` \
	        --nnodes $SLURM_JOB_NUM_NODES \
			--nproc_per_node $SLURM_GPUS_PER_NODE \
			--rdzv_id $SLURM_JOB_ID \
			--rdzv_backend c10d \
			--rdzv_endpoint $master_node:29500 \
				c_training/lvae_train.py

sleep 1
cd $current_dir
mv slurm-${jobid}.out $outdir
