import extended_state as gm

agents = [gm.Agent(gm.Rewarder(), 'snake1_single.pth')]

gm.train(agents)