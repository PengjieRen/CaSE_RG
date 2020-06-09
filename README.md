# CaSE-1.0
 Conversations with Search Engines
 
## 1. Config environment.
+ conda create --name python3.7 python=3.7
+ source ${HOME}/.bashrc
+ conda activate python3.7
+ pip install -r requirements.txt
 
## 2. Prepare data. The format of each file is as follows:
+ .split (train/dev/test split)
```
#query_id	#split
71_1	train
36_4	dev
25_8	test
```
+ .answer (summarized answers)
```
#context_id	#query_id	#passage_id	#answer
17_1;17_2;17_3	17_4	CAR_45c972f40634d3ce1ec65743db8612908f251db9	The mechanical energy is the energy associated with the motion and position of an object. It is the sum of potential energy and kinetic energy.
```
+ .passage (passage collection)
```
#passage_id	#passage
MARCO_5941958	Lewis and Clark's expedition traveled more often by boat via rivers than by land, and this route follows the rivers as closely as possible. Occasional rough roads, narrow to nonexistent shoulders, and sparse services make this one of our more challenging routes.
```
+ .support (supporting spans used to generate summarized answers)
```
#context_id	#query_id	#passage_id	#support	#support	#support	#support
17_1;17_2;17_3	17_4	CAR_45c972f40634d3ce1ec65743db8612908f251db9	In the physical sciences, mechanical energy is the sum of potential energy and kinetic energy. It is the energy associated with the motion and position of an object.
```
+ .pool (candidate passage pool)
```
#query_id #Q0 #passage_id #rank #score #model
1_1 Q0 MARCO_955948 1 2.846400022506714 Anserini
```
+ .qrel (passage relevance labels)
```
#query_id #0 #passage_id #relevance
1_1 0 MARCO_955948 2
```
+ .query (conversational query)
```
#query_id	#query
1_2	What are the educational requirements required to become one?
```
+ .reformulation.query (complete reformulated query)
```
#query_id	#reformulated_query
1_2	What are the educational requirements required to become a physician's assistant?
```
## 3. Run Prepare_dataset.py to process the data.
After running the scripts, you should get 1) '.pkl' files corresponding to raw files; 2) '.dataset_name.train/dev/test.model_name.dataset.pkl' files for each model and each dataset.
 
## 4. Do training or inference: 
+ ./run.sh model_name train/test slurm_node data_path dataset_name
 
## 5. Do evaluation:
+ ./evaluate.sh model_name data_path
 
