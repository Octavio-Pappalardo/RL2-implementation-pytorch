import torch
import numpy as np


########------------------------LIFETIME BUFFER---------------------###########

#Buffer for collecting all the lifettime's data in an inner loop

#------------------------------------------------------------------------------

'''When deciding what to store at position t in the lifetime buffer there is are at least two options. Here it was decided to store at 
position t the last information needed for the agent to make the prediction a_t ; this simplifies how the inputs to the rnn at each timestep are obtained and
the making of predictions during running the inner loop . The tradeoff is that this means storing at position t information
about the timestep t-1 (like a_{t-1}) ; this makes the loss calculations and advantage estimation for the outer loop update a bit less easy to read because it means certain information
belonging to a certain timestep is displaced by one.

At position t it is stored: (s_t ,a_{t-1} , logprob_{t-1}, d_{t-1} ,r_{t-1}  , metaValue_t ,metaAdvantage_t ,metaReturnsToGo_t) .
A dummy a,logprob,d,r is stored at position 0 and a dummy s is stored in the last position'''


class Lifetime_buffer:
    def __init__(self,num_lifetime_steps , env, device,env_name='none'):
        '''class for storing all the data the agent collects throughout an inner loop . It is used in outer loop updates'''

        self.observations=torch.zeros( (num_lifetime_steps+1 , env.observation_space.shape[0])).to(device)
        self.prev_actions= torch.zeros((num_lifetime_steps+1, env.action_space.shape[0])).to(device)
        self.prev_logprob_actions= torch.zeros((num_lifetime_steps+1)).to(device)
        self.prev_rewards= torch.zeros((num_lifetime_steps+1)).to(device)
        self.dones= torch.zeros((num_lifetime_steps+1)).to(device)
        #they have size num_lifetime_steps +1 because we actually want to store 1 extra step (refer to the discussion above)

        self.meta_values= torch.zeros((num_lifetime_steps)).to(device)

        #observations[i] stores the state at which the agent was in step i
        #prev_actions[i] stores the action the agent took in step i-1
        #prev_logprob_actions[i] stores the log probabilty prev_actions[i] had of being taken
        #prev_rewards[i] stores the reward the agent got for doing prev_actions[i] at observation[i-1] - the reward obtained in timestep i-1
        #meta_values[i] stores the state meta value estimate (an estimate of the expected meta-return to go) the agent estimated taking into account all lifetime up to and including observation[i]
        #Dones[i] stores whether when taking step[i-1] the env was terminated or tuncated.
        #in other words, it says wether the env was reset befor step i. In which case observation[i] is the first obs from the new episode

        self.meta_advantages=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_returns_to_go=torch.zeros((num_lifetime_steps)).to(device)


        self.device=device

        self.num_lifetime_steps=num_lifetime_steps
        self.episodes_returns=[]  #list that contains the returns of each episode in the lifetime
        self.episodes_successes=[] #list that contains wether each episode in the lifetime succeded in completing the task 

        self.env_name=env_name

    def store_step_data(self,global_step, obs, prev_act, prev_reward, prev_logp,prev_done):
        self.observations[global_step]=obs.to(self.device)
        self.prev_actions[global_step]=prev_act.to(self.device)
        self.prev_logprob_actions[global_step]=prev_logp.to(self.device)
        self.dones[global_step]=prev_done.to(self.device)
        self.prev_rewards[global_step]=prev_reward.to(self.device)

    def store_meta_value(self,global_step, meta_value):
        self.meta_values[global_step]=meta_value

    def preprocess_data(self, data_stats , objective_mean):
        ''' Normalizes rewards using environment dependant statistics. It multiplies the e_rewards by a factor that makes the mean equal to objective_mean
        Args:
            data_statistics : An object that keeps track of the mean extrinsic reward given by each environment type
            objective_mean : The --approximate-- mean reward per step after normalization
        '''
        self.normalized_prev_rewards= (self.prev_rewards.clone().detach() / (data_stats.e_rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean



    def compute_meta_advantages_and_returns_to_go(self,gamma=0.95, e_lambda=0.95):
        self.calculate_returns_and_advantages_with_standard_GAE(prev_objective_rewards=self.normalized_prev_rewards,gamma=gamma,gae_lambda=e_lambda)
      
    def calculate_returns_and_advantages_with_standard_GAE(self,prev_objective_rewards,gamma=0.99,gae_lambda=0.95):
        lastgaelam = 0
        for t in reversed(range(self.num_lifetime_steps)):
            if t == self.num_lifetime_steps - 1:
                nextnonterminal = 0.0 
                nextvalue = 0.0
            else:
                nextnonterminal = 1.0 
                nextvalue = self.meta_values[t + 1]
            delta = prev_objective_rewards[t+1] + gamma * nextvalue * nextnonterminal - self.meta_values[t]
            self.meta_advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        self.meta_returns_to_go = self.meta_advantages + self.meta_values


#########----------------------------OUTER LOOP BUFFER---------------------##############

#Buffer for combining data from multiple inner loops and updating the outer loop agent

#----------------------------------------------------------------------------------------


class OL_buffer:
    def __init__(self,device):
        #Initialize space for storing data for a batch. Batch of data to update the meta agent.
        #After applying self.combine_data() , each column becomes data from a different step and each column data from a different environment
        #for elements that are vectors and not scalars they go into a 3rd dimension.
        self.num_lifetimes = 0

        self.observations = []
        self.prev_actions = []
        self.prev_logprob_actions = []
        self.dones = []
        self.prev_rewards= []

        self.meta_values= []
        self.meta_returns_to_go = []
        self.meta_advantages = []


        self.device=device

    def collect_lifetime_data(self, lifetime_buffer):
        # Append data from a given lifetime_buffer to the combined buffer
        self.num_lifetimes += 1

        self.observations.append(lifetime_buffer.observations)
        self.prev_actions.append(lifetime_buffer.prev_actions)
        self.prev_logprob_actions.append(lifetime_buffer.prev_logprob_actions)
        self.dones.append(lifetime_buffer.dones)
        self.prev_rewards.append(lifetime_buffer.prev_rewards)
   
        self.meta_values.append(lifetime_buffer.meta_values)
        self.meta_returns_to_go.append(lifetime_buffer.meta_returns_to_go)
        self.meta_advantages.append(lifetime_buffer.meta_advantages) 


    def combine_data(self):
        #Stack the data from each lifetime into single tensors 
        self.observations=torch.nn.utils.rnn.pad_sequence(self.observations, batch_first=False,padding_value=0.0).to(self.device)
        self.prev_actions=torch.nn.utils.rnn.pad_sequence(self.prev_actions, batch_first=False,padding_value=0.0).to(self.device)
        self.prev_logprob_actions=torch.nn.utils.rnn.pad_sequence(self.prev_logprob_actions, batch_first=False,padding_value=0.0).to(self.device)
        self.dones=torch.nn.utils.rnn.pad_sequence(self.dones, batch_first=False,padding_value=0.0).to(self.device)
        self.prev_rewards=torch.nn.utils.rnn.pad_sequence(self.prev_rewards, batch_first=False,padding_value=0.0).to(self.device)


        self.meta_values=torch.nn.utils.rnn.pad_sequence(self.meta_values, batch_first=False,padding_value=0.0).to(self.device)
        self.meta_returns_to_go=torch.nn.utils.rnn.pad_sequence(self.meta_returns_to_go, batch_first=False,padding_value=0.0).to(self.device)
        self.meta_advantages=torch.nn.utils.rnn.pad_sequence(self.meta_advantages, batch_first=False,padding_value=0.0).to(self.device)



    def clean_buffer(self):
        self.num_lifetimes = 0
        self.observations = []
        self.prev_actions = []
        self.prev_logprob_actions = []
        self.dones = []
        self.prev_rewards= []

        self.meta_values= []
        self.meta_returns_to_go = []
        self.meta_advantages = []

