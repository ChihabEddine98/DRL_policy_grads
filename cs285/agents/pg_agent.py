import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        """ 
            TODO #1 ✅ : update the PG actor/policy using the given batch of data
            ---------------------------------------------------------------------
            d[J_θ] = E[ Σ d[log(π(a|s))] * (Σ r(𝛕) -b)] 
            

        """
        
        """ step ⓵  :  get Q(s_t, a_t) using rewards (r_0, ..., r_t, ..., r_{T-1}) """
        q_values = self.calculate_q_vals(rewards_list)

        """ step ⓶  : get A(s_t, a_t) = Q(s_t,a_t) - V(s_t) """
        advantages = self.estimate_advantage(observations, q_values,terminals)

        """ step ⓷  : update PG """
        train_log = self.actor.update(observations, actions, advantages,q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        """

            TODO #2 ✅ : return the estimated qvals based on the given rewards, using
                        full trajectory-based estimator or RTG estimator
            ---------------------------------------------------------------------
            Return : Q^{π}(s_t, a_t)  
        """
        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(r) for r in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            """
                TODO #3 ✅ : values were trained with standardized q_values
                ----------------------------------------------------------------------
                 The predictions have the same mean and standard deviation as
                 the current batch of q_values
            """
            values = values_unnormalized * np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                """
                     TODO #4 ✅ : recursively compute advantage
                     -----------------------------------------
                            estimates starting from timestep T.
                """
                for i in reversed(range(batch_size)):
                    if terminals[i] != 1:
                        advantages[i] = rews[i]+self.gamma*values[i+1]-values[i]
                        advantages[i] += self.gamma*self.gae_lambda*advantages[i+1]
                    else:
                        advantages[i] = rews[i]-values[i]
                        
                advantages = advantages[:-1]

            else:
                """ TODO #5 ✅ : compute advantage estimates using q_values, and values as baselines """
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            """ TODO #6 ✅ : standardize the advantages to have a mean of zero and a standard deviation of one """
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        """
                TODO #7 ✅: create list_of_discounted_returns
                ---------------------------------------------
        """
        discounted_returns = np.array([(self.gamma**t)*rewards[t] for t in range(len(rewards))])
        return [np.sum(discounted_returns)]*len(rewards)

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        """
            TODO #8 ✅ : create `list_of_discounted_returns`
        """
        
        T = len(rewards)
        gammas = self.gamma ** np.arange(0, T)
        list_of_discounted_cumsums = np.zeros(T)
        
        for t in range(T):
            list_of_discounted_cumsums[t] = np.sum(gammas[:T-t] * rewards[t:])

        return list_of_discounted_cumsums
    

