using Gym.Main.Space:Box, sample
using LyceumBase, LyceumMuJoCo, MuJoCo, Shapes, LyceumMuJoCoViz

mutable struct AntEnv <: AbstractMuJoCoEnvironment
    model_path::String
    sim::MJSim
    observation_space::Box
    action_space::Box
    state::AbstractVecOrMat
    skip_frame::Integer
end

function init!(n::Integer=1)
    model_path = joinpath(@__DIR__, "ant.xml")
    #Load xml file
    sim = MJSim(model_path, skip = n)
    #Define Space
    obs_high = ones((sim.m.nq - 2 + sim.m.nv + sim.m.nbody * 6,))*(1e6)
    action_high = ones((sim.m.nu,))*(1e6)
    observation_space = Box(-obs_high, obs_high, Float64)
    action_space = Box(-action_high, action_high, Float64)
    state = zeros((sim.m.nq+sim.m.nv,))
    skip_frame = n
    #Create env
    env = AntEnv(model_path, sim, observation_space, action_space, state, skip_frame)
    reset!(env)
    return env
end

function setstate(env::AntEnv)
    #Set state
    temp = env.sim.d
    return vcat(temp.qpos,temp.qvel)
end

function getobs(env::AntEnv)
    #Get obs
    return vcat(env.sim.d.qpos[3:length(env.sim.d.qpos)], env.sim.d.qvel, Iterators.flatten(clamp.(env.sim.d.xfrc_applied,-1,1)))
end

function forward!(env::AntEnv, action)
    #Apply the action and move forward by n=env.skip_frame frames
    env.sim.d.ctrl[:] = action
    for _ in range(1,stop=env.skip_frame)
        LyceumBase.step!(env.sim)
    end
end

@inline torso_x(env::AntEnv) = env.sim.d.qpos[1]

function step!(env::AntEnv, action)
    #step function
    @assert action ∈ env.action_space "$action in $(env.action_space) invalid"
    xposbefore = torso_x(env)
    forward!(env, action)
    xposafter = torso_x(env)
    forward_reward = (xposafter-xposbefore)/(env.skip_frame)
    ctrl_cost = 0.5 * sum(action.^2)
    contact_cost = 0.5 * 1e-3 * sum(clamp.(env.sim.d.xfrc_applied,-1,1).^2)
    survive_reward = 1.0
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    env.state[:] = setstate(env)
    done = !(all(isfinite.(env.state)) && env.state[3] >=0.2 && env.state[3] <=1.0)
    ob = getobs(env)
    return ob, reward, done, Dict()
end

function reset!(env::AntEnv)
    #reset function
    LyceumBase.reset!(env.sim)
    ob = getobs(env)
    return ob
end

Base.show(io::IO, env::AntEnv) = print(io, "AntEnv")

#Test Script
env = init!()
T=convert(Int64, 100)
actions = [sample(env.action_space) for i=1:T]
states = Array(undef, statespace(env.sim), T)
i = 1
done = false
reset!(env)
while i <= length(actions) && !done
    global i, done
    a, b, done, d = step!(env, actions[i])
    states[:, i] .= LyceumBase.getstate(env.sim)
    i+=1
end

#Script to open visualizer
#BUG: find how to make script run the episode
LyceumMuJoCoViz.visualize(env.sim, trajectories=[states],
    controller=LyceumBase.setaction!(
        env.sim, sample(env.action_space))
    )