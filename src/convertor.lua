require 'cunn'
require 'cutorch'
require 'NeuralQLearner'
require 'Scale'
require 'TransitionTable'
require 'Rectifier'
require 'lfs'

_dir = 'agents/size5_focus/'
for file in lfs.dir(_dir) do
  print(file)
  if lfs.attributes(_dir .. file,"mode") == "file" and file == "EP40000S753928" then
    print("yes")
    local agent_learnt=torch.load(_dir .. file)
    agent_learnt.network:float()
    local agent = {network = agent_learnt.network, ep = agent_learnt.ep, numSteps = agent_learnt.numSteps}
    torch.save(_dir .. file .. '-converted', agent)
  end
  collectgarbage()
end
