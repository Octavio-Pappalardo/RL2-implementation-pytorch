import numpy as np
import torch
import wandb
from torch.utils.data import BatchSampler, SubsetRandomSampler 


# ------------- Logger . For logging metrics ---------------
class Logger:
    def __init__(self , num_epsiodes_of_validation=2):
        '''
        num_epsiodes_of_validation : sets how many of the last episodes of each lifetime are used for computing the validation metrics 
        '''
        self.lifetimes_mean_episode_return= [] #stores the mean episode return of the lifetimes used during the outer loop training
        self.lifetimes_success_percentage =[] #for all lifetimes used during the outer loop training ,it stores the percentage of episodes where the agent succedded in the lifetime

        self.per_env_total_return={}  #stores the total return of the lifetimes but for each different type of env used (instead of a single list for all envs)
        self.per_env_success_percentage={}
        self.last_episode_return=[]  #stores the return in the last episode of each lifetime
        self.last_episode_success_percentage=[]

        self.validation_episodes_return=[]  #stores the return in the last 'num_epsiodes_of_validation' episodes of each lifetime
        self.validation_episodes_success_percentage=[]
        self.num_epsiodes_of_validation=num_epsiodes_of_validation


    def collect_per_lifetime_metrics(self, lifetime_buffer):
        self.lifetimes_mean_episode_return.append(np.array(lifetime_buffer.episodes_returns).mean())
        self.lifetimes_success_percentage.append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))
        
        self.last_episode_return.append(lifetime_buffer.episodes_returns[-1])
        self.last_episode_success_percentage.append(lifetime_buffer.episodes_successes[-1])

        self.validation_episodes_return.append( np.array(lifetime_buffer.episodes_returns[-self.num_epsiodes_of_validation:]).mean() )
        self.validation_episodes_success_percentage.append(np.mean(lifetime_buffer.episodes_successes[-self.num_epsiodes_of_validation:]) )

        if lifetime_buffer.env_name not in self.per_env_total_return:
            self.per_env_total_return[f'{lifetime_buffer.env_name}']= []
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}']= []
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))
        else:
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))


    def log_per_update_metrics(self , num_inner_loops_per_update):
        #log per environment metrics
        for env_name in self.per_env_total_return:
            env_return= np.array(self.per_env_total_return[env_name][-10:]).mean()
            env_success= np.array(self.per_env_success_percentage[env_name][-10:]).mean()
            wandb.log({env_name+' returns': env_return ,env_name+' success':env_success}, commit=False)
       
        #log metrics taking a mean over all the lifetimes considered for the update
        last_episode_return=np.array(self.last_episode_return[-num_inner_loops_per_update:]).mean()
        last_episode_success_percentage=np.array(self.last_episode_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'last episode return': last_episode_return ,'last episode success percentage':last_episode_success_percentage}, commit=False)

        validation_episodes_return=np.array(self.validation_episodes_return[-num_inner_loops_per_update:]).mean()
        validation_episodes_success_percentage=np.array(self.validation_episodes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'validation episodes return': validation_episodes_return ,'validation episodes success percentage':validation_episodes_success_percentage}, commit=False)


        mean_episode_return=np.array(self.lifetimes_mean_episode_return[-num_inner_loops_per_update:]).mean()
        lifetime_success_percentage=np.array(self.lifetimes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'mean episode return': mean_episode_return ,'lifetime success percentage':lifetime_success_percentage})




#--------Samplers (for sampling tasks) ------------------- 


def Sampler(items, batch_size):
    '''given a list it creates an iterator that yields random batches of elements from the list
    Args:
        items : The list of items to sample from
        batch_size : the number of items to yield each time next() is called
    '''
    #if batch_size > than the number of elements to sample from then each element should be at least Q times in the batch where Q is batch_size//len(items)
    if batch_size >len(items):
        base_batch= items * (batch_size//len(items))
        effective_batch_size= batch_size-len(base_batch) #the number of elements we actually need to sample
    else:
        base_batch=[]
        effective_batch_size=batch_size

    #If the requested batch_size is a multiple of the number of available items , simply return copies of all items until the batch is filled (no sampling needed)
    if effective_batch_size==0:
        while True:
            yield base_batch


    indices=np.arange(len(items))
    sampler = SubsetRandomSampler(indices)
    batch_sampler = BatchSampler(sampler, batch_size=effective_batch_size, drop_last=True)
    
    while True:
        for indices in batch_sampler:
            batch = [items[i] for i in indices]
            yield batch+base_batch


def Tasks_batch_sampler(benchmark,batch_size):
    '''Samples a batch of tasks for ML10 and ML50 benchmarks.
    Args:
        benchmark : benchmark to sample from
        batch_size : The number of tasks to sample

    Yields batches of tasks
    '''
    envs_in_benchmark= [name for name,env_cls in benchmark.train_classes.items()]
    env_type_sampler=Sampler(envs_in_benchmark,batch_size)

    task_samplers={name :Sampler([task for task in benchmark.train_tasks if task.env_name == name] ,1) for name,enc_cls in benchmark.train_classes.items()}
    while True:
        sampled_env_types=next(env_type_sampler) #sample what envs will go in the batch (there could be repetition if batch_size>len(benchmark.train_classes))
        sampled_tasks=[next(task_samplers[f'{env_name}']) for env_name in sampled_env_types] #for each sampled environment sample a task
        sampled_tasks=[task for task_in_list in sampled_tasks for task in task_in_list] #just formating; so that batch doesnt have each element inside its own list.
        yield sampled_tasks



#---------- Statistics tracker  ----------
#keeps track of reward statistics for normalization puproses

class Statistics_tracker:
    def __init__(self ):
        self.e_rewards_means={}  #keeps track of the mean extrinsic reward given by each environment type
        self.e_rewards_vars={}

        #for calculating a running mean of extrinsic rewards
        self.num_lifetimes_processed={}
        self.means_sums={}

    def update_statistics(self,lifetime_buffer):
        ##update rewards statistics
        
        sample_mean= torch.mean(lifetime_buffer.prev_rewards[1:])
        
        #first time that environment type is encountered
        if lifetime_buffer.env_name not in self.e_rewards_means:
            self.e_rewards_means[f'{lifetime_buffer.env_name}']= sample_mean 
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']=1
            self.means_sums[f'{lifetime_buffer.env_name}'] =sample_mean
        
        else:
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']+=1
            self.means_sums[f'{lifetime_buffer.env_name}'] += sample_mean
            self.e_rewards_means[f'{lifetime_buffer.env_name}']= self.means_sums[f'{lifetime_buffer.env_name}'] / self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']

    