import wandb

api = wandb.Api()

# list of runs ids
path = 'dewald123/liu_pytorch_cnn/'
id_dict= {0:[], 1:[]}
id_dict[0] = ['bms1w9fy', 'd9ries1s']
id_dict[0] = ['bms1w9fy', 'd9ries1s']
runs_dict = {0:[],1:[]}

# Project is specified by <entity/project-name>
run = api.run(f"{path}{id_dict[0][0]}")
if run.state == "finished":
   for i, row in run.history().iterrows():
      print(row["_step"], row["Validation_TSS"])
