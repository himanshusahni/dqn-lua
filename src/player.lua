visualizers = {}
require 'qt'
require 'qttorch'
require 'torch'
require 'image'

local player = torch.class('visualizers.Player')
function player:__init(game_env)
  self.game_env = game_env    -- set the game to be played
  self.game_actions = self.game_env:getActions()
end

function player:greedy(network, state)
  q = network:forward(state):float():squeeze()
  local maxq = q[1]
  local besta = {1}
  -- Evaluate all other actions (with random tie-breaking)
  for a = 2, table.getn(self.game_actions) do                                                -- picking the best q and action
      if q[a] > maxq then
          besta = { a }
          maxq = q[a]
      elseif q[a] == maxq then
          besta[#besta+1] = a
      end
  end
  self.bestq = maxq                                                           -- recording the best q on this step
  local r = torch.random(1, #besta)                                           -- random tie breaking happening
  self.lastAction = besta[r]                                                  -- random tie breaking
  return q, besta[r]
end

--[[give the user the option to play the game. if agent is given, it'll display
the agent's best move as well.]]
function player:play(agent)
  while true do
    self.game_env:reset()
    local screen, reward, terminal = self.game_env:getState()    -- grab a screen
    w = image.display{image = screen, zoom = 8}
    -- query user for action till they chose to exit
    while (not terminal) do
      os.execute("clear")
      io.write("Last Reward = " .. reward .. "\n")
      if agent then
        local state = torch.FloatTensor(1, 1, screen:size(2), screen:size(3))
        state[{1, {}, {}, {}}] = image.rgb2y(screen)
        agentPreferences, besta = self:greedy(agent.network, state)
      end
      io.write("press 1-12 for actions\n" .. self:actionExplanation(self.game_actions, agentPreferences))
      if besta then
        io.write("Agent epsilon is at " .. agent.ep .. " at " .. agent.numSteps .. " steps \n")
        io.write("Agent prefers action " .. besta .. ". " .. self.game_actions[besta] .. "\n")
      end
      i = io.read("*n")   -- get action from user
      if i <= 12 and i > 0 then
        self.game_env:step(self.game_actions[i])    -- step through legal actions
      else break end
      screen, reward, terminal = self.game_env:getState()    -- grab another screen and repaint image
      local qtimg = qt.QImage.fromTensor(screen)
      w.painter:image(0,0, screen:size(2)*8, screen:size(3)*8,qtimg)
      collectgarbage()
    end
    if i > 12 or i < 0 then
      break
    end
    print("reward" .. reward)
    io.read("*n")
  end
end

--[[prints action choices and if agent is given, its preference]]
function player:actionExplanation(actions, agentPreferences)
  local s = ""
  for action_num, action in ipairs(actions) do
    s = s .. action_num .. ". " .. action
    if agentPreferences then
      s = s .. "    " .. string.format("%.3f" , agentPreferences[action_num])
    end
    s = s .. "\n"
  end
  return s
end

package.path = package.path .. ";/Users/home/research/deep-learning/domain/trash-collector/src/?.lua"
require 'trashCollector'
require 'NeuralQLearner'
require 'Scale'
require 'TransitionTable'
require 'Rectifier'
p = visualizers.Player(domain.TrashCollector(require('params')))
-- agent = torch.load('agents/EP16500S495983-converted')
-- p:play(agent)
p:play()
