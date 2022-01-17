import simple_game as gm

agents = [gm.Agent(gm.Rewarder(closer_to_food=0), 'snake3_single.pth')]

gm.train(agents)