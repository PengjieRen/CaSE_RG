# CaSE-1.0
 Conversations with Search Engines
 
## 1. Config environment.
+ conda create --name python3.7 python=3.7
+ source ${HOME}/.bashrc
+ conda activate python3.7
+ pip install -r requirements.txt
 
## 2. Prepare data. "./dataset" constains some demo data files.
+ .split
#query_id	#split
71_1	train
36_4	train
25_8	test
 
## 3. Run Prepare_dataset.py to process the data.
 
## 4. Do training or inference: 
+ ./run.sh model_name train/test slurm_node data_path dataset_name
 
## 5. Do evaluation:
+ ./evaluate.sh model_name data_path
 
