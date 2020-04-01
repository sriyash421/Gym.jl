using Flux, Gym
using Flux.Optimise: Optimiser
using Statistics: mean, norm
using DataStructures: CircularBuffer
using Distributions: sample
using Printf
using CuArrays
using Random

env = make("CartPole-v0", :human_pane)
reset!(env)


STATE_SIZE = length(env._env.state)
ACTION_SIZE = length(env._env.action_space)
γ = 0.9
MAX_NUM_STEPS = 10000
MAX_EPSIODES = 5000
η = 3e-4

model = Chain(Dense(STATE_SIZE, 128, relu),Dense(128, ACTION_SIZE)) |> gpu
opt = Optimiser(Adam(η))

function get_action (state)
    global model
    global ACTION_SIZE
    probs = model(state |> gpu)
    highest_prob_action = sample(Array(range(0,length=ACTION_SIZE)),Weights(probs) |>cpu) 
    log_prob = log(probs[highest_prob_action])
    return highest_prob_action, log_prob
end

function update_policy(rewards, log_probs)
    discounted_rewards = []
    for i in range(0,length=length(rewards))
        Gt = 0f0
        pw = 0
        for r in rewards[i:]
            Gt += (γ^pw)*r
            pw += 1
        end
        push!(discounted_rewards, Gt)
    end
    discounted_rewards = (discounted_rewards - mean(discounted_rewards))/(norm(discounted_rewards))
    score = sum(-log_probs.*discounted_rewards)
    Flux.train!((), params(model),score , opt)
end

function episode!(train=true, draw=false)
    done=false
    rewards = []
    log_probs = []
    t = 0
    while !done && t<MAX_NUM_STEPS
        draw && render!(env)
        s = env._env.state
        action, log_prob = get_action(s)
        s, reward, done, _ = step!(env, action)
        train && push!(rewards, reward)
        train && push!(log_probs, log_prob)
        draw && sleep(0.01)
        train && done && update_policy(rewards, log_probs)
        t+=1
    end
    t == MAX_NUM_STEPS && !done && update_policy(rewards, log_probs)
    return sum(rewards)
end
# -----------------------------------------
e = 1
scores = CircularBuffer{Float32}(100)

while true
  global e
  reset!(env)
  total_reward = episode!(env)
  push!(scores, total_reward)
  print("Episode: $e | Score: $total_reward ")
  last_100_mean = mean(scores)
  print("Last 100 episodes mean score: $(@sprintf "%6.2f" last_100_mean)")
  if last_100_mean > 195
    println("\nCartPole-v0 solved!")
    break
  end
  e += 1
end

