--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
dqn = {}
local nql = torch.class('dqn.NeuralQLearner')
--[[ DEUBUGGING
require 'qt'
require 'qttorch'
require 'torch'
require 'image'
--]]
function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.                   -- same as number of total pixels = 7056
    self.actions    = args.actions                                              -- actions coming from the environment's getActions method
    self.n_actions  = #self.actions                                             -- number of actions
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1                                              -- epsilon greedy stuff. args.ep was 1.
    self.ep         = self.ep_start -- Exploration probability.                 -- epsilon starts off as 1
    self.ep_end     = args.ep_end or self.ep                                    -- ep_end is defined as 0.1
    self.ep_endt    = args.ep_endt or 1000000                                   -- ep_endt is 1 mil

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.                      -- lr is 0.00025
    self.lr             = self.lr_start                                         -- the current lerning rate
    self.lr_end         = args.lr_end or self.lr                                -- le_end is not given
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1                              -- 32
    -- self.valid_size     = args.valid_size or 500                                -- 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.              -- 0.99
    self.update_freq    = args.update_freq or 1                                 -- 4, number of steps after which a minibatches is run
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1                                    -- 1, number of minibatches run above
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0                                 -- 50000
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000                         -- 1 mil
    self.hist_len       = args.hist_len or 1                                    -- 4,
    self.rescale_r      = args.rescale_r                                        -- 1
    self.max_reward     = args.max_reward                                       -- 1
    self.min_reward     = args.min_reward                                       -- -1
    self.clip_delta     = args.clip_delta                                       -- 1
    self.target_q       = args.target_q                                         -- 10,000
    self.bestq          = 0

    self.gpu            = args.gpu                                              -- -1 or 1

    self.ncols          = args.ncols or 1  -- number of color channels in input -- 1
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 35, 35} -- 4*1x84x84= 4x84x84
    self.full_dims      = args.full_dims or {self.hist_len*self.ncols, 35, 35}
    self.focus_x        = args.focus_x                                          -- focus power
    self.focus_y        = args.focus_y
    self.focus          = args.focus                                            -- the environment decides where the focus begins (usually in the center)
    self.preproc        = args.preproc  -- name of preprocessing network        -- net_downsample_2x_full_y
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1                                 -- nothing so 1
    self.bufferSize     = args.bufferSize or 512                                -- 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()                      -- args.network is our convnet_atari3
    self.transfer   = args.transfer
    -- first check if a transfer network exists
    if self.transfer then
      local msg, err = pcall(torch.load, self.transfer)   -- load the transfer agent in protected mode
      if not msg then
        error("transfer network does not exist")
      end
      print("Loading transfer network from " .. self.transfer)
      self.network = err.network
    else
      -- check whether there is a network file
      local msg, err = pcall(require, self.network)
      if not msg then
        error("the network does not exist")
      end
      print('Creating Agent Network from ' .. self.network)
      self.network = err
      self.network = self:network()
    end
    collectgarbage()    -- clear any temp variables while loading network
    -- Load preprocessing network.
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- now center the focus
    self:reFocus()

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,                 --84x84, game gives number of actions by getActions (this is length of that table)
        histLen = self.hist_len, gpu = self.gpu,                                -- 4, 0 or 1
        maxSize = self.replay_memory, histType = self.histType,                 -- 1000000, linear,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,         -- 1, 1,
        bufferSize = self.bufferSize                                            -- 512
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.                            -- starting number of steps should be 0
    self.lastState = nil                                                        -- nothing observed yet
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()                              -- gives all weights and gradients of them in a single tensor
    self.dw:zero()                                                              -- zero out the gradients

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then                                                       -- it is 10,000
        self.target_network = self.network:clone()
    end
end

function nql:reFocus()
  self.focus = {x = self.full_dims[2]/2, y = self.full_dims[3]/2}
end

function nql:setFocus(focus)
  self.focus = focus
end

function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:clone():float(), self.focus):reshape(self.state_dim)
    end
    return rawstate
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)              -- this is just the discounted reward. If terminal = 1, the second
    term = term:clone():float():mul(-1):add(1)                                  -- term is 0 so it'll just be r.

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network                                      -- yes, target_q is 10000
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
    q2_max = target_q_net:forward(s2):float():max(2)                            -- running s2 through the NN and finding out maximum value.

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then                                                      -- yes
        delta:div(self.r_max)                                                   -- r_max = 1
    end
    delta:add(q2)                                                               -- added it to r

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()                               -- getting q
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)                                                            -- delta term should be complete now

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta                      -- just clipping deltas
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]                                             -- only the a[i] values are set, everything else remains 0
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)      -- picking out 32 transitions randomly from data so far.
    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,        -- calulcates the discounted reward term (r + blah)
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()                                                              -- done everytime

    -- get new gradient
    self.network:backward(s, targets)                                           -- calculates gradients according to input

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)                                    -- just getting learning rates

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)                                         -- 0.95*g + 0.05*gradients
    self.tmp:cmul(self.dw, self.dw)                                             -- squaring dw
    self.g2:mul(0.95):add(0.05, self.tmp)                                       -- 0.95*g2 + 0.05*dw*dw
    self.tmp:cmul(self.g, self.g)                                               -- g^2
    self.tmp:mul(-1)                                                            -- -(g^2)
    self.tmp:add(self.g2)                                                       -- -(g^2) + g2
    self.tmp:add(0.01)
    self.tmp:sqrt()                                                             -- sqrt(-(0.95*g + 0.05*dw)^2 + 0.95*g2 + 0.05*dw^2)

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)                      -- self.dw/self.tmp * self.lr
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)          -- this is where the RL happens
    -- Preprocess state (will be set to nil if terminal)                        -- the rawstate is just the frame (no history)
    local state = self:preprocess(rawstate):float()                 -- crop it down to 84x84
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)                              -- 1 if reward is > 1
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)                              -- -1 if reward is < -1
    end
    if self.rescale_r then                                                      -- true
        self.r_max = math.max(self.r_max, reward)                               -- again max between 1 and reward
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()
    --Store transition s, a, r, s'
    if self.lastState and not testing then                                      -- nill to start out with
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    -- if self.numSteps == self.learn_start+1 and not testing then                 -- 50,001
    --     self:sample_validation_data()                                           -- just fills up validation sars
    -- end

    curState= self.transitions:get_recent()

    curState = curState:resize(1, unpack(self.input_dims))                      -- input_dims is 4x84x84

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)                        -- evaluates the current state, picks best action and does epsilon greedy
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and                     -- after 50,000 steps
        self.numSteps % self.update_freq == 0 then                              -- every 4 steps
        for i = 1, self.n_replay do                                             -- just 1
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal

    if self.target_q and self.numSteps % self.target_q == 1 then                -- after every target_q steps (10000), the target network changes to the current network
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex
    else
        return 0                                                               -- if the state was terminal then no action is required
    end
end

                                                                                -- pure epsilon greedy method. anneals Epsilon every time it is called if not testing
function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.

    if state:dim() == 2 then                                                    -- state has to be 3d
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()                     -- q values for all actions
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do                                                -- picking the best q and action
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

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
