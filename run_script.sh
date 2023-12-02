
#!/bin/sh
### General options
### –- specify queue -- gpua100, gpuv100, gpua10, gpua40
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30
# request 5GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo joboutput/gpu.out
#BSUB -eo joboutput/gpu.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.9.10
module load cuda/11.6
source "${HOME}/Deep-Learning/.venv/bin/activate"

# Exit if previous command failed
if [[ $? -ne 0 ]]; then
	exit 1
fi

# run training
# python3 unet_train_noise.py
python3 VGG_correct_try_pnt.py
