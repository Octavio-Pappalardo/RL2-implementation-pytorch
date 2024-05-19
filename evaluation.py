
class Evaluation_buffer:
    def __init__(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]

    def collect_data(self, lifetime_return_per_episode ,lifetime_success_per_episode):
        self.lifetimes_returns_per_episode.append(lifetime_return_per_episode)
        self.lifetimes_success_per_episode.append(lifetime_success_per_episode)

    def combine_data(self):
        self.lifetimes_returns_per_episode=np.array(self.lifetimes_returns_per_episode)
        self.lifetimes_success_per_episode=np.array(self.lifetimes_success_per_episode)
        # after running each should be a numpy array of size (num_lifetimes , num_episodes_per_lifetime)

    def clean_buffer(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]



def collect_evaluation_data(model_path, n , config_setting , benchmark_name='ML1' ,env_name= 'door-close-v2',
                            run_deterministically=True , evaluate_test_tasks=True):
    ''' For a given benchmark and environment name it collects all the test(or train) tasks corresponding to that environment ,
    evaluates the performance of the meta agent when being applied to those tasks and returns data of that evaluation.

    Args:
        model_path : path to the weights of the model to be evaluated
        n : amount of times each evaluation task is considered
        config_setting: configurations . should match those used when training the agent to be evaluated so that it can be loaded properly
        benchmark_name : name of the benchmark to evaluate on .
        env_name : name of the environment to evaluate on (just applies if benchmark_name='ML1').
        run_deteministically: wether the meta agent is runned in a deterministic (choosing most probable action) or probabilistic manner 
        evaluate_test_tasks : if false the evaluation is done over the tasks with which the meta agent trained instead of with the test tasks

    Returns:
        lifetimes_returns_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing the return of each episode
        lifetimes_success_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing whether each episode succeded
    '''
    from config import get_config
    from inner_loop import run_inner_loop
    import ray

    if ray.is_initialized:
        ray.shutdown()
    ray.init()
    config=get_config(config_setting)

    # Construct the benchmark and get all the evaluation tasks (50)
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{env_name}', seed=config.seed)
        if evaluate_test_tasks==True:
            eval_tasks= benchmark.test_tasks
        else:
            eval_tasks= benchmark.train_tasks

    elif benchmark_name=='ML10' or benchmark_name=='ML45':
        benchmark = metaworld.ML10(seed=config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=config.seed)
        if evaluate_test_tasks==True:
            eval_tasks=[task for task in benchmark.test_tasks
                        if task.env_name == env_name]
        else:
            eval_tasks=[task for task in benchmark.train_tasks
                        if task.env_name == env_name]

    eval_tasks=eval_tasks * n

    eval_buffer=Evaluation_buffer() 


    remote_inner_loop=ray.remote(run_inner_loop)
    meta_agent_training=False
    inputs= [ ((config,model_path,benchmark_name,task),meta_agent_training,run_deterministically ) for task in eval_tasks]
    results =ray.get([remote_inner_loop.options(num_cpus=1).remote(*args) for args in inputs])

    for episodes_returns,episodes_successes in results:
        eval_buffer.collect_data(lifetime_return_per_episode=episodes_returns ,lifetime_success_per_episode=episodes_successes )

    eval_buffer.combine_data()

    return eval_buffer.lifetimes_returns_per_episode , eval_buffer.lifetimes_success_per_episode




if __name__=='__main__':
    from config import get_config
    import metaworld

    import re
    import os
    import numpy as np
    import time

    ###define parameters of the evaluation run ###
    evaluate_test_tasks=True    #wether to perform the evaluation over the training or test tasks
    model_path= "./rl2_models/ML1_soccer-v2__1701257845__best_model.pth"
    benchmark_name='ML1' 
    ml1_env_name= 'soccer-v2' #only relevant when using ML1 benchmark
    config_setting= 'metaworld'
    n= 5
    run_deterministically=True
    ###  ----- ----- ###

    
    # Extract the time from the file name (identifies the model) using regular expression . assumes model_path has a format like "../../models/example123__1638439876__best_model.pth"
    model_id = re.search(r'(?<=__)\d+', os.path.basename(model_path)).group()

    #get a list of the names of all the environments that will be evaluated (each of them contains 50 tasks with parametric variation)
    if benchmark_name=='ML1':
        envs_to_evaluate=[ml1_env_name]
    else:
        ol_config=get_config(config_setting)
        benchmark = metaworld.ML10(seed=ol_config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=ol_config.seed)
        if evaluate_test_tasks==True:
            envs_to_evaluate= [name for name,env_cls in benchmark.test_classes.items()]
        else:
            envs_to_evaluate= [name for name,env_cls in benchmark.train_classes.items()]

    for env_name in envs_to_evaluate:
        start_time=time.time()

        exp_name= f'{benchmark_name}_{env_name}'
        run_name = f"{exp_name}__{model_id}" 
        returns_data_path = f"./rl2_eval_data/{run_name}__eval_returns.npy"
        successes_data_path = f"./rl2_eval_data/{run_name}__eval_successes.npy"

        lifetimes_returns_per_episode_data ,lifetimes_success_per_episode_data= collect_evaluation_data(model_path=model_path , 
                                                            n=n , config_setting=config_setting ,
                                                            benchmark_name=benchmark_name ,env_name= env_name,
                                                            run_deterministically=run_deterministically, evaluate_test_tasks=evaluate_test_tasks)
        

        np.save(returns_data_path ,lifetimes_returns_per_episode_data)
        np.save(successes_data_path ,lifetimes_success_per_episode_data )

        print(f'evaluation data of experiment : {run_name} ready')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data[:,-4:]) }')
        print(f'time take in minutes ={(time.time()-start_time)/60}')