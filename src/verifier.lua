package.path = package.path .. ";../../domain/trash-collector/src/?.lua"
require 'trashCollector'
require 'NeuralQLearner'
require 'Scale'
require 'TransitionTable'
require 'Rectifier'
require 'lfs'
require 'image'

local verifier = torch.class('Verifier')
function verifier:__init(game_env, opt)
  self.game_env = game_env(opt)    -- set the game to be played
  self.game_actions = self.game_env:getActions()
  self.n_actions = #self.game_actions
  self.valid_steps = opt.valid_steps
  self.ep = opt.valid_ep
  self.gpu = opt.gpu
end

--[[picks the best action greedily given a state and a network]]
function verifier:greedy(network, state)
  if self.gpu and self.gpu >= 0 then
    state = state:cuda()
  end
  q = network:forward(state):float():squeeze()
  local maxq = q[1]
  local besta = {1}
  -- Evaluate all other actions (with random tie-breaking)
  for a = 2, self.n_actions do   -- picking the best q and action
      if q[a] > maxq then
          besta = { a }
          maxq = q[a]
      elseif q[a] == maxq then
          besta[#besta+1] = a
      end
  end
  self.bestq = maxq   -- recording the best q on this step
  local r = torch.random(1, #besta)   -- random tie breaking happening
  self.lastAction = besta[r]    -- random tie breaking
  return besta[r]
end

--[[picks the epsilon greedy action given a state and a network]]
function verifier:eGreedy(network, state)
  if torch.uniform() < self.ep then   -- if below epsilon then random action
      return torch.random(1, self.n_actions)
  else
      return self:greedy(network,state)
  end
end


--[[does verfication runs for an agent]]
function verifier:play(agent)
  if self.gpu and self.gpu >= 0 then
    agent.network = agent.network:cuda()
  end
  local step
  local avg_reward = 0
  for step = 1, self.valid_steps do
    self.game_env:reset()   -- start a new game
    print("Starting game no. " .. step .. " for agent " .. agent.numSteps)
    local screen, reward, terminal = self.game_env:getState()    -- grab a screen
    local ep_reward = reward
    while (not terminal) do
      local state
      if self.gpu and self.gpu >= 0 then
        state = torch.CudaTensor(1, 1, screen:size(2), screen:size(3))
      else
        state = torch.FloatTensor(1, 1, screen:size(2), screen:size(3))
      end
      state[{1, {}, {}, {}}] = image.rgb2y(screen:clone():float())
      local besta = self:eGreedy(agent.network, state)   -- feed the agent the state and pick eGreedy action
      self.game_env:step(self.game_actions[besta])    -- step through action
      screen, reward, terminal = self.game_env:getState()    -- grab next screen
      ep_reward = ep_reward + reward
    end
    avg_reward = avg_reward + ep_reward
  end
  avg_reward = avg_reward/self.valid_steps    -- get averaged reward
  print("Average reward for agent " .. agent.numSteps .. " is " .. avg_reward)
  return avg_reward
end

-- set up options
local opt = require('params')
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

-- create verifier
v = Verifier(domain.TrashCollector, opt)
_dir = 'agents/one_quick/'
out_file = io.open(_dir .. 'results.txt', 'w')

-- run!
results = {}
for file in lfs.dir(_dir) do
  if (lfs.attributes(_dir .. file, "mode") == "file") and (file ~= "results") and (file ~= "results.txt") then
    local agent = torch.load(_dir .. file)
    if agent.numSteps > 200000 then
      local avg_reward = v:play(agent)
      results[agent.numSteps] = avg_reward
      out_file:write(agent.numSteps, " ", avg_reward, "\n")
      out_file:flush()
    end
  end
  collectgarbage()
end

io.close(out_file)
torch.save(_dir .. 'results', results)


for i,r in pairs(results) do
  -- print("Steps: " .. i .. " Average Reward: " .. r)
  print(i .. " " .. r)
end
