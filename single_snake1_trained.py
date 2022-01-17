import simple_game as gm

agents = [gm.Agent(gm.Rewarder(), 'snake1_single.pth', load_from_model='snake1_single.pth')]

gm.train(agents, random_rounds=0)