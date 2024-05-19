class Metaworld:
    def __init__(self):
        self.benchmark_name= 'ML1'  #'ML10' , 'ML45'
        self.env_name= 'door-close-v2' 

        self.num_epsiodes_of_validation = 2
        self.num_lifetimes_for_validation = 60

        self.seeding=False
        self.seed=1
        self.ol_device='cuda'
        self.il_device='cpu'

        self.num_outer_loop_updates=5000 
        self.num_inner_loops_per_update = 30
        self.num_il_lifetime_steps=4500  
        

        self.learning_rate=5e-4
        self.adam_eps=1e-5

        self.rewards_target_mean= 0.1 
        self.meta_gamma= 0.995 
        self.bootstrapping_lambda=0.95

        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 256 
        self.initial_std=1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 10, 
            'num_minibatches': 0,
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.005,
            "valuef_coef": 0.5,
            "clip_grad_norm": True, 
            "max_grad_norm": 0.5,
            "target_KL": 0.1
        }



def get_config(config_settings):
    if config_settings=='metaworld':
        return Metaworld()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    



##################-------- HYPERPARAMETERS DETAILS ------------##################


#####-----  Dataset-benchmark used -----####
#benchmark_name : Selects which benchmark to run
#env_name  : selects choice of environment when benchmark name is 'ML1' . EJ  'pick-place-v2', 'door-close-v2' , 'soccer-v2'


######----- model validation performance -----#########
#num_epsiodes_of_validation : controls how many of the last episodes of each lifetime are used for computing the vaildation metrics
#num_lifetimes_for_validation : controls how many of the last lifetimes are used to compute the validation metrics
          #note that validation metrics are just the metrics used for deciding which model version to keep but they arent computed on a different dataset


#ol_device : controls on what device the meta agent is updated 
#il_device : controls on what device the meta agent runs while executing the inner loops 


####----- amount of data -------########
#num_outer_loop_updates  : number of times to collect data from inner loops and update the reward agent
#num_inner_loops_per_update  number of inner loops runned to collect data for the outer loop update
#num_il_lifetime_steps  :  how many steps are executed in each inner loop 


####----learning rate -------#####
#learning_rate: optimizers learning rate
#adam_eps  : adam epsilon parameter

####---- advantage estimation ------########
#rewards_target_mean  :  the extrinsic rewards from each  environment are normalized such that their new mean value is e_rewards_target_mean
#meta_gamma  : the gamma considered for the meta objective 
#bootstrapping_lambda : lambda used for advantage estimation

###-----RL2 agent parameters----#### 
#rnn_input_size  : size of the imput to the rnn at each step  
#rnn_type  :  The type of rnn network used ,'gru' ,'lstm' ,'rnn'            
#rnn_hidden_state_size : The dimensionality of the rnn hidden state
#initial_std :  the initial standard deviation of each dimension of the actions given b the agent


#####---TBPTT PPO parameters----#####
#k  :  when using TBPTT determines the length at which gradients are cuttof. k=k1=k2 .
#update_epochs  :        #number of epochs runned inside the PPO update
#num_minibatches    :      the number of mini-batches created from each batch for the PPO update - number of gradient step taken per epoch . If 0 one is taken for each k length sequence.
#normalize_advantage    :  Toggles advantages normalization -  if True the advantages of the whole batch are normalized
#clip_coef  :              the PPO clipping coefficient 
#entropy_coef   :     coefficient of the entropy - controls the weight of the exploration encouraging entropy loss in the total loss with wich agent is updated.
#valuef_coef    :     coefficient of the value function - controls the weight of the value function loss in the total loss with wich agent is updated.
#max_grad_norm  :     the maximum norm the gradient of each gradient step is allowed to have. Parameter for gradient clipping
#target_KL  :      the KL divergence threshold