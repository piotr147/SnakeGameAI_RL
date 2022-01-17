import simple_game as gm

agents = [gm.Agent(gm.Rewarder(), 'snake1_single.pth')]

gm.train(agents)