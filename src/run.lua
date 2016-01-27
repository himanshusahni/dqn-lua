package.path = package.path .. ";../../domain/trash-collector/src/?.lua"

require 'torch'
require 'trashCollector'
require 'nn'
require 'image'
require 'NeuralQLearner'
require 'TransitionTable'
require 'Rectifier'
require 'paths'

happyEndings = 0    -- DEBUGING
--[[ run a game till termination, can be used for validation runs as well
returns the total episode rewards and number of steps taken
]]
local function runTillTerminate(game_env, agent, valid)
  local episode_reward = 0
  local episode_steps = 0
  local action_index = 0
  local game_actions = game_env:getActions()
  local screen, reward, terminal = game_env:getState()    -- gets the first frame

  while true do
    -- feed the frame to the agent and learn
    if valid then
      action_index = agent:perceive(reward, screen, terminal, true, opt.valid_ep)   -- validation run, no learning going on
    else
      action_index = agent:perceive(reward, screen, terminal)   -- training run, learning happens
    end
    if reward == 1 then
      happyEndings = happyEndings + 1
    end
    if not terminal then
      screen, reward, terminal, new_focus = game_env:step(game_actions[action_index], agent.focus)
      if new_focus then
        agent.setFocus(new_focus)   -- in case look action just set the new focus
      end
      episode_steps = episode_steps + 1
      episode_reward = episode_reward + reward
    else
      break   -- if this state was terminal, this episode is done
    end
  end
  return episode_reward, episode_steps
end

-- set up options
local opt = require('params')
-- before creating agent, we must set GPU (so that the net is created appropriately)
if opt.gpu and opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.gpu == 0 then
        local gpu_id = tonumber(os.getenv('GPU_ID'))
        if gpu_id then opt.gpu = gpu_id+1 end
    end
    if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
    opt.gpu = cutorch.getDevice()
    print('Using GPU device id:', opt.gpu-1)
else
    opt.gpu = -1
      print('Using CPU code only.')
end

--first thing is to create the agent, game environment and game_actions
local game_env = domain.TrashCollector(opt)
opt.actions = game_env:getActions()
-- now create agent
local agent = dqn.NeuralQLearner(opt)

-- set up a few things
steps= 0   -- number of steps in game
local episode_reward = 0    -- reward earned this episode
episode_num = 0

--let's go!
while(steps < opt.steps) do
  game_env:reset()
  episode_num = episode_num + 1
  print("episode " .. episode_num .. " steps " .. steps .. " epsilon " .. agent.ep .. " successes " .. happyEndings)  -- DEBUGGING
  -- check if we need to validate
  if episode_num%opt.episodes == 0 then
    torch.save(opt.agentdir .. 'EP40000S' .. steps, agent)
    torch.save('numhappyEndings', happyEndings)
    os.exit()
  end
  -- now run the training episode
  episode_reward, episode_steps = runTillTerminate(game_env, agent)
  steps = steps + episode_steps
  -- collect garbage every 100 episodes
  if episode_num%100 == 0 then collectgarbage() end

  -- save rewards and model after every few times
  if episode_num%opt.save_freq == 0 then
    torch.save(opt.agentdir .. 'EP' .. episode_num .. 'S' .. steps, agent)
  end
end
torch.save(opt.agentdir .. 'EP' .. episode_num .. 'S' .. '2000000', agent)
torch.save('numhappyEndings', happyEndings)
os.exit()
