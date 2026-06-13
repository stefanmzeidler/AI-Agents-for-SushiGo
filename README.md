**AI Agents for SushiGo!**
Stefan Zeidler
Department of Computer Science
CS-710 Artificial Intelligence
Dr. Susan McRoy
May 19, 2025

# Introduction

I am terrible at card games. My best friend is excellent at card games. In fifteen years, I haven’t beaten him at any card game. This is mostly due to both a lack of short-term memory and patience. However, AI agents excel in both of these areas. That is why I decided to create an AI agent to play SushiGo! using Monte Carlo Tree Search (MCTS) to rectify this situation. Additionally, I used machine learning to train other agents using self-play and compared their performance to the MCTS agent.

I will first briefly go over a basic explanation of the SushiGo! game and Monte Carlo Tree Search, and what makes this approach AI. Then I will introduce the reinforcement learning approach that was used, results of both agent types versus a random player, and against each other. I will discuss these results, as well as limitations, and avenues for further research. At several different points, I also consulted ChatGPT for help, and I note where this was done.

# Background

## SushiGo!

SushiGo! is a multi-player card game in which players attempt to build the highest scoring selection of cards. There are 12 card types, and different combinations of these cards yield different point values. Each turn, a player selects a card from their hand and then passes the remaining cards to the next player. The game lasts for three rounds, and scores are tallied at the end of each round. There are also special cards that are only scored at the very end of the game or round, and the player with the most gains points, while the player with the least loses points.

Interestingly, this game can be divided into two sections with imperfect and perfect information respectively. In the initial stages of each round, players do not know which cards are in play until they have had every other player’s hand passed to them. At this point, all the cards in play have been seen by each player, and the game now can be considered to theoretically have perfection information, even if practically it’s hard to memorize that many different cards.

There are also a special “Chopsticks” card with more complicated play which was not implemented for the purposes of this project. Unfortunately, the logic proved challenging to implement and I feel that the game is sufficient for demonstrating the capability of AI agents without this feature. However, in the future, it would be very interesting to see how both AI agent types handle this card.

## Game Playing and Adversarial Search

In terms of artificial intelligence, games can be conceived of as multi-agent environments in which the other agents work to defeat each other, including the AI agent(s) in question. One of the main ways that adversarial agents are modeled is using a tree search. For example, in minmax search, moves by each player are given utility values according to some evaluation function, and values from terminal states are backed up towards the root, allowing an agent make effective choices (Russell et al., 2021).

However, this method suffers from complexity caused by depth and branching. As the number of choices increases and the length of a game increases, the performance of a tree search can suffer as more and longer paths from root to terminal nodes must be considered. Several strategies have been used to address this, including pruning branches and cutting off searches early and the use of heuristic functions (Russell et al., 2021).

However, sometimes the branching factor and length cannot easily be dealt with, and it may also be difficult to determine an effective heuristic function. A specific kind of tree search, Monte Carlo Tree Search, overcomes these difficulties by conducting several simulations of complete games, and evaluates the average utility (win percentage) of moves from a specified starting state. Because MCTS is promising for games such as SushiGo!, I elected to use this for my agent.

However, to be effective, a good selection policy that balances exploration and exploitation is needed (Russell et al., 2021)

## UCB1

UCB1 is an effective selection policy that balances both of these factors, as evaluated by Powel et al. in their 2013 paper comparing MCTS selection policies. UCB1 is a formula that ranks moves based on a combination of the average utility of going through that move’s node in the tree plus an exploration term that decreases as a node has been visited more often. The balance between exploitation and exploration represented by these two terms can be modified by a tunable constant (Powley et al., 2013). It is for this reason that I selected UCB1 as the selection policy for my MCTS agent.

Although MCTS and UCBS are effective searches, they are not the current state-of-the-art for AI agents and game playing. Instead of tuning a parameter, agents can instead determine an effective playout policy through reinforcement learning.

## Reinforcement learning

In reinforcement learning, an agent makes or is given an observation of the relevant environment. It then selects an action and is given a corresponding reward from the environment (Wei Wei, 2022). In the context of games, through repeated play, the agent develops a policy to select the moves that yield that most rewards. One of the strategies for training an agent is training using a Deep Q-Network (DQN), which was selected for this project and is discussed in more detail below.

# Design

For my overall design, I created an environment that simulates and manages the game, and to which I can supply different player agents to evaluate their effectiveness. Since the main game environment is fairly simple, I will focus my discussion on the players and the environment used to train the D

## RandomPlayer

As a baseline, I created a RandomPlayer that selects a random move each time. This player helps ascertain whether other models perform better than random chance.

## MCTSPlayer

The MCTS agent simulates playouts at every turn, the number of which can be modified through parameters. The balance between exploration and exploitation can also be modified through a c\_param, but this value was kept constant at 1.4 given the high win performance exhibited by the agent, shown in the results section. The agent works in conjunction with an MCTSNode class that conducts that actual simulation and returns the best move according to its selection policy.

By default, this selection policy is UCB1 and discussed above. However, as Aleks discussed in his presentation, having an overbearing opponent that one has little chance of beating does not make for fun gameplay (even if it means I can beat my friend at games).

UCB1 always selects the node (action) with the highest utility. Essentially, a standard MCTS agent will always choose the best move that it can find. To make an MCTS more fair to play against, Demediuk et al. propose different mechanisms for selecting a node after search has been completed. Among these is a method called Proactive Outcome-Sensitive Action Selection (POSAS). In this selection method, an interval around zero is specified in which all actions have the same utility and then one of these actions is randomly selected. This prevents the MCTS agent from always selecting the absolute best, but should still yield better performance than random chance by still disallowing the worst moves – those outside the interval (Demediuk et al., 2017). In their evaluation, they selected an interval based on absolute percentage since they were trying to target health points, a fixed value, in their game. In contrast, the points scored in SushiGo! vary from round to round and game to game, so I decided to use an interval based on score variance based on suggestions provided by ChatGPT. The considers all scores within one standard distribution of the average as equal.

## DQN Agent

To develop the DQN agent, I followed a tutorial provided by TensorFlow. The Jupyter Notebook version can be found [here](https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb) (TF-Agents Authors, 2023). The DQN agent is trained using self-play to develop a selection policy for moves.

The DQN agent uses a neural network consisting of several fully connected dense layers, the last of which contains a separate neuron for each output. The agent passes its observation of the environment into the network and returns a selection based on which neuron in the final layer was activated. Each neuron also contains weights on the output going from one node to the next, which are adjusted during training. The agent aims to reduce the difference between its predicted reward and an actual reward using a loss function. In this case, the Mean Squared Error (MSE) was used as the loss function. The neurons in each layer also use the rectified linear activation function (ReLu), even though it may lead to dead neurons (Kelleher et al., 2020). Both leaky relu and sigmoid activation functions were tried, but they yielded results equivalent to and worse than random respectively.

Although SushiGo! occurs in three rounds, with points awarded at the end of each round, rewards were not given to the agent until the very end of the game. This is because there are several cards that award points only at the end of the game, and because the desired state was to win the game, not necessarily each individual round. Attempting to score the highest on each round may not work due to the cards that span the entire game. This does mean, however, that rewards are not immediate, and delayed for however long the game lasts.

One challenge in designing this agent was the fact that a fixed number of actions must be specified at the start, and agent will select from these actions. In this case, the actions are choosing from one of the eleven cards. However, the range of *valid* actions is dynamic – the player will not always have one of each card in their hand. As the game goes on, the number of cards in each player’s hand decreases, and duplicate cards are considered the same action. To address this, I needed to create an action mask that prevents the agent from selecting invalid cards to play. Since this was not discussed in the tutorial and the documentation that I could find did not provide examples of masks, I needed some help from ChatGPT to create the mask.

For training the DQN Agent, the game environment was modified to match the requirements of the PyEnvironment class from TensorFlow. Both the DQN agent and environment have modifiable hyperparameters such as the number of dense layers, the neurons in each layer, and the number of iterations for training. The results can be found in the corresponding results section later in this paper.

## User Interface

For the sake of transparency, I will also address how the user interface was created. Unfortunately, I haven’t had the opportunity to take a UI design course yet and have very little experience creating one. The game logic was implemented and could be played through a command line interface. However, two considerations made me decide to try to create a UI for the game. First, my friend does not know how to use command line interfaces and playing a card game is difficult if you can’t see the cards. Second, a command line interface does not make for a compelling presentation in class.

This means the UI was almost wholly developed by ChatGPT. It did not alter the game logic, but I provide my files to it and it created a UI that would represent the game logic already in use. I also was unable to find digital images of the cards, so I needed to use photographs. I unfortunately did not discover the tabletop games framework, which includes SushiGo, until after I had already finished this portion.

# Relation to Artificial Intelligence

Both of these agents can be considered artificial intelligence since they follow the PEAS acronym. They both use a Performance measure to evaluate their choices, one being the error between predicted and actual rewards, and the other being the average utility of different actions.

They operate within an Environment, which they interact with using Actuators (choosing a card) and Sensors (game state information). Choosing an action transitions states, and the agents are working towards a goal state (winning the game).

# Methodology

For both of the agents, I compared different versions using different parameters against two random players in the evaluation environment. The agents would play 100 games against the random players and at the end the average score and win percentage would be calculated. Additionally, the number of times that each agent placed first, second or third was recorded, as well as the standard deviation of their scores.

The parameters that were changed for the MCTS agent were the selection policy, whether UCB1 or POSAS, and the number of simulations the agent would use in its tree search. The higher the number of simulations the better the performance at playing the game, but at the cost of more time needed per turn.

The DQN agent and environment can be changed to use different numbers of layers and neurons in each layer. Due to poor performance of sigmoid and leaky relu activation functions, and unfamiliarity with other activation functions, the relu activation function was used.

The best performing DQN agent was then placed in the evaluation environment to play against a UCB1 MCTS player and a random player. The results of all these trials are show in the in next section.

# Results and Discussion

With the exception of playing against my friend, for each of these trials, 100 games were conducted and the average score, standard deviation, win rate, and times placed first, second, and third were recorded. For the games against my friend, only the MCTS agent was tested, and only five games were played (I do not think he would like trying to play 100 games).

## MCTS agent versus friend

| Player | Wins |
| --- | --- |
| Friend | 3 |
| MCTS | 2 |

As stated in my presentation, I would consider this a success even though my friend won more games. This is because these are my first wins against him, even if I wasn’t technically the one playing.

My friend was able to use strategies that sabotaged the MCTS player by preventing sets and giving it cards that did not contribute points.

## MCTS Agent vs Random

| Player | Avg Score | Std Deviation | Win Rate | 1st/2nd/3rd |
| --- | --- | --- | --- | --- |
| MCTS | 52.18 | 7.16 | 95% | 95, 5, 0 |
| Random1 | 32.,24 | 7.96 | 2% | 2, 51, 47 |
| Random2 | 30.71 | 9.31 | 3% | 3, 44, 53 |

The results of these trials show that an MCTS agent that always selects the best movie does indeed perform better than random – significantly so. It won 95/100 games and placed second for the five games it did not win. It did not place last at all.

## MCTS Agent using POSAS vs Random

| Player | Avg Score | Std Deviation | Win Rate | 1st/2nd/3rd |
| --- | --- | --- | --- | --- |
| MCTS | 41.83 | 7.64 | 53% | 53,30,17 |
| Random1 | 37.28 | 9.74 | 29% | 29,34,37 |
| Random2 | 34.75 | 8.65 | 18% | 18,36,46 |

When the POSAS method for selecting actions is applied to the MCTS agent, the win rate drops to 53%. This shows that, while it still performs better than random chance, it is not oppressively difficult to overcome. This indicates that POSAS does work as a method of adjusting the difficulty of an MCTS agent.

## DQN Agent Results

In the following tables, I show the results of DQN agents with different hyper parameters versus two random bots. To save space, the average scores, standard deviation, and placements have been omitted. These can be provided upon request or new trials can be generated from the program.

| Layers | Neurons per Layer | Training Iterations | Activation Function | Win Rate |
| --- | --- | --- | --- | --- |
| 3 | (100, 100, 100) | 400 | Sigmoid | 13% |
| 3 | (100, 100, 100) | 400 | LeakyRelu | 31% |
| 3 | (100, 100, 100) | 3000 | Relu | 74% |
| 3 | (200, 200, 200) | 1000 | Relu | 54% |
| 3 | (200, 200, 200) | 2000 | Relu | 83% |
| 3 | (200, 200, 200) | 4000 | Relu | 67% |
| 1 | (100) | 3000 | Relu | 80% |
| 4 | (50, 50, 50, 50) | 3000 | Relu | 20% |
| 4 | (100,100,100,100) | 3000 | Relu | 26% |
| 2 | (100, 100) | 3000 | Relu | 94% |
| 2 | (50, 50) | 3000 | Relu | 16% |
| 2 | (200,200) | 3000 | Relu | 93% |
| 2 | (300, 300) | 3000 | Relu | 98% |
| 2 | (400, 400) | 3000 | Relu | 41% |

Due to time constraints, only a few samples of varying parameters were taken. Nevertheless, the results still do indicate regions where underfitting and overfitting may occur. For instance, given the three layer Relu agents, they appear to be underfitting prior to 2000 training iterations, then overfitting any noise in the gameplay after 3000 iterations.

Likewise, the best performance was achieved using two layers, with underfitting with fewer layers and overfitting with more layers. The maximum win rate occurred at 300 neurons per layer with two layers, with underfitting and overfitting with less or more neurons per layer.

The best performing agent had two layers, with 300 neurons per layer, and this was pitted against both the normal MCTS agent and the POSAS MCTS agent.

Best DQN Agent vs normal MCTS vs Random

| Player | Avg Score | Std Deviation | Win Rate | 1st/2nd/3rd |
| --- | --- | --- | --- | --- |
| DQN | 44.72 | 8.70 | 34% | 34,59,7 |
| MCTS | 50.32 | 6.52 | 66% | 66,34,0 |
| Random | 25.28 | 6.62 | 0% | 0, 7, 93 |

When playing against an MCTS agent, the DQN agent’s performance drastically declined, and it won only about a third of the time. However, the MCTS agent also performed worse, indicating that it does not have perfect gameplay. With better tuned hyper parameters, it is possible that the DQN agent could beat the MCTS agent.

Best DQN Agent vs POSAS MCTS vs Random

| Player | Avg Score | Std Deviation | Win Rate | 1st/2nd/3rd |
| --- | --- | --- | --- | --- |
| DQN | 52.89 | 10.31 | 87% | 87, 12, 1 |
| MCTS Posas | 39.38 | 7.11 | 10% | 10, 77, 13 |
| Random | 28.69 | 7.14 | 3% | 3, 11, 86 |

When playing against the dynamic difficulty versions of the MCTS agent, the DQN agent won a very high percentage of the time. The MCTS agent came in second a higher percentage of the time, as would be expected if it were to lose against the DQN agent but still perform better than average. This also indicates that the difficulty is indeed being modulated, but perhaps by too much against the DQN agent. By selecting different intervals for equality, performance could be adjusted. The random bot somehow won 3 matches, possibly with some very lucky selections.

# Conclusions

Given the above information, I believe it would be safe to say that MCTS agents are effective at playing SushiGo!, but are vulnerable to exploitation by skilled human players. Furthermore, the difficulty of these agents can modified, but this parameter needs to be tuned.

DQN agents show promise at playing SushiGo!, and to further improve the agents I would conduct a more thorough analysis of the hyperparameters to narrow down the best combination. I believe there are also ways to for the agents themselves to learn these parameters. I would like to conduct a larger number of trials to highlight the regions of underfitting and overfitting for each parameter to find where the peak(s) are.

I would also like to see how the DQN agent plays against humans. It may have a lower win rate against the MCTS agent, but it also may be less vulnerable to being exploited.

Works Cited

Demediuk, S., Tamassia, M., Raffe, W. L., Zambetta, F., Li, X., & Mueller, F. (2017). Monte Carlo tree search based algorithms for dynamic difficulty adjustment. *2017 IEEE Conference on Computational Intelligence and Games (CIG)*, 53–59.

Kelleher, J. D., Mac Namee, B., D’Arcy, A., & ProQuest (Firm). (2020). Fundamentals of machine learning for predictive data analytics: Algorithms, worked examples, and case studies. In *Fundamentals of machine learning for predictive data analytics: Algorithms, worked examples, and case studies* (Second edition.). MIT Press.

Powley, E. J., Whitehouse, D., & Cowling, P. I. (2013). Bandits all the way down: UCB1 as a simulation policy in Monte Carlo Tree Search. *2013 IEEE Conference on Computational Inteligence in Games (CIG)*, 1–8.

Russell, S. J. (Stuart J., Norvig, P., Chang, M.-W., Devlin, J., Dragan, A., Forsyth, D., Goodfellow, I., Malik, J., Mansinghka, V., Pearl, J., & Wooldridge, M. J. (2021). Artificial intelligence: A modern approach / Stuart J. Russell and Peter Norvig?; contributing writers: Ming-Wei Chang, Jacob Devlin, Anca Dragan, David Forsyth, Ian Goodfellow, Jitenda M. Malik, Vikash Mansinghka, Judea Pearl, Michael Wooldridge. In *Artificial intelligence: A modern approach* (Fourth edition.). Pearson.

TF-Agents Authors. (2023). *Train a Deep Q Network with TF-Agents | TensorFlow Agents*. TensorFlow. https://www.tensorflow.org/agents/tutorials/1\_dqn\_tutorial

Wei Wei (Director). (2022, February 10). *Reinforcement learning overview (Reinforcement learning with TensorFlow Agents)* [Video recording]. TensorFlow. https://www.youtube.com/watch?v=52DTXidSVWc
