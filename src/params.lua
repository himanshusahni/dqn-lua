-- all network and agent options go in here

opt = {}
-- domain options
opt.width = 5   -- gridworld width
opt.height = 5    -- gridworld height
opt.ptu = 7   -- pixels per gridworld unit
opt.num_trash = 1   -- number of trash items to be picked up

-- agent-domain options
opt.focus_x = 35    -- width of focus area (for now equal to pixels in image)
opt.focus_y = 35    -- height of focus area
-- opt.focus_std = 2*opt.ptu   -- pixels after which focus drops by 1 std
-- opt.focus_jump = opt.ptu    -- pixels by which the focus jumps around
opt.state_dim = opt.focus_x*opt.focus_y
opt.ncols = 1   -- number of color channels in input
opt.hist_len = 1    -- number of frames per state
opt.input_dims = {opt.hist_len*opt.ncols, opt.focus_x, opt.focus_y}   -- input to the convnet first layer
opt.full_dims = {opt.hist_len*opt.ncols, opt.width*opt.ptu, opt.height*opt.ptu}
opt.rescale_r = 1   -- whether or not to rescale rewards
opt.max_reward = 1    -- if rescaling rewards, what is the maximum reward
opt.min_reward = -1    -- minimum reward


--[[ agent options ]]--
opt.ep = 1    -- starting epsilon
opt.ep_end = 0.1    -- final epsilon
opt.ep_endt = 500000   -- number of steps after which epsilon stops annealing
opt.valid_ep = 0.05   -- epsilon for validation runs
opt.lr = 0.001    -- learning rate
opt.discount = 0.99   -- discount factor

--[[algorithm options]]--
opt.minibatch_size = 32   -- size of minibatch training NN
opt.update_freq = 1   -- number of steps after which a minibatches is run
opt.n_replay = 1    -- number of minibatches run above
opt.learn_start = 10000   -- Number of steps after which learning starts.
opt.bufferSize = 512   -- some buffer in transition function
opt.replay_memory = 1000000   -- size of the memory
opt.hist_len = 1    -- number of frames per state
opt.clip_delta = 1    -- whether to clip the gradients in backprop
opt.target_q = 5000    -- how often to update the target q network
opt.input_dims = {opt.hist_len*opt.ncols, opt.focus_x, opt.focus_y}   -- input to the convnet first layer
opt.preproc = "net_downsample_2x_full_y"    -- preprocessing neural network, for now should do nothing
opt.network = "convnet_atari3"
-- opt.transfer = 'agents/one_two/EP40000S1355909-converted'

--[[running options]]
opt.gpu = 1   -- GPU
opt.steps = 2000000    -- maximum number of steps to take
opt.episodes = 40000    -- maximum number of episodes to run
opt.save_freq = 500   -- model is saved after these many steps
opt.valid_freq = 40000   -- number of EPISODES after which the model is validated
opt.valid_steps = 25    -- number of EPISODES validation is averaged overq
opt.modelname = "models/convnet_atari3"
opt.agentdir = "agents/one_quick/"
return opt
