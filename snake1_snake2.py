import simple_game as gm

agents = [gm.Agent(gm.Rewarder(), 'snake1_single.pth'), gm.Agent(gm.Rewarder(further_from_food=0, closer_to_food=0), 'snake2_single.pth')]

gm.train(agents)