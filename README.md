# MONTE CARLO CONTROL ALGORITHM

## AIM
To implement the Monte Carlo Control algorithm to optimize decision-making in a reinforcement learning environment by approximating the optimal action-value function.

## PROBLEM STATEMENT
The problem is to determine an optimal policy for a given environment where the dynamics are unknown. The environment presents stochastic rewards and state transitions, and the task is to maximize cumulative rewards over time by improving the policy. The Monte Carlo Control algorithm achieves this by learning from sampled episodes to evaluate and improve policies..

## MONTE CARLO CONTROL ALGORITHM
step 1
Initialize the action-value function 𝑄(𝑠,𝑎)arbitrarily for all state-action pairs, and initialize a policy 𝜋 based on 𝑄(𝑠,𝑎).

step 2
Generate an episode using the current policy π, which consists of a sequence of state-action-reward tuples

step 3
Compute the return 𝐺𝑡(cumulative reward from time 𝑡).

step 4
Update 𝑄(𝑠𝑡,𝑎𝑡) by averaging the observed returns.

step 5
Improve the policy 𝜋(𝑠)=argmax 𝑄(𝑠,𝑎). The Monte Carlo control function updates the action-value function Q by averaging returns. It can be expressed as: ![image](https://github.com/user-attachments/assets/7d67acda-0223-4f88-b7a2-376adbc6f1bc)
image where α is a learning rate, and G is the return (cumulative reward). This function helps in refining Q(s,a) values, which subsequently improves the policy for maximizing the expected reward.

## MONTE CARLO CONTROL FUNCTION
```
def mc_control (env, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 5000, max_steps = 300, first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n

  #Write your code here
  discounts=np.logspace(0,max_steps,num=max_steps,base=gamma,endpoint=False)
  alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
  epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
  pi_track=[]
  Q=np.zeros((nS,nA),dtype=np.float64)
  Q_track=np.zeros((n_episodes,nS,nA),dtype=np.float64)

  select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

  for e in tqdm(range(n_episodes),leave=False):
    trajectory=generate_trajectory(select_action,Q,epsilons[e],env,max_steps)
    visited=np.zeros((nS,nA),dtype=bool)
    for t,(state,action,reward,_,_) in enumerate(trajectory):
      if visited[state] [action] and first_visit:
        continue
      visited[state][action]=True
      n_steps=len(trajectory[t:])
      G=np.sum(discounts[:n_steps]*trajectory[t:,2])
      Q[state][action]=Q[state][action]+alphas[e]*(G-Q[state][action])
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
  V=np.max(Q,axis=1)
  pi=lambda s: {s:a for s,a in enumerate (np.argmax(Q,axis=1))}[s]
  #return Q, V, pi, Q_track, pi_track
  return Q, V, pi


optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes = 3000)
print('Name:jayasri                  Register Number:  212222240028      ')
print_state_value_function(optimal_Q, P, n_cols=4, prec=2, title='Action-value function:')
print_state_value_function(optimal_V, P, n_cols=4, prec=2, title='State-value function:')
print_policy(optimal_pi, P)



print('Name: jayasri                   Register Number: 212222240028        ')
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))
```

## OUTPUT:
### Name:Jayasri dodda
### Register Number:212222240028

![image](https://github.com/user-attachments/assets/d8b3c45f-fe21-4bdd-9777-64b71b1f7eaf)



![image](https://github.com/user-attachments/assets/185bc7ce-b435-4a06-bad7-ad0174d1378f)



## RESULT:

Thus implement the Monte Carlo Control algorithm to optimize decision-making in a reinforcement learning environment by approximating the optimal action-value function done sucessfully.
