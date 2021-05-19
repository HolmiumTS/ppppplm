import matplotlib.pyplot as plt
import json

f = open("../../trainer_state.json", 'r')
data = json.loads(f.read())
train_x = []
train_y = []
eval_x = []
eval_y = []

for record in data['log_history']:
    if 'eval_loss' in record:
        eval_x.append(record['step'])
        eval_y.append(record['eval_loss'])
    else:
        train_x.append(record['step'])
        train_y.append(record['loss'])

plt.plot(train_x, train_y)
plt.plot(eval_x, eval_y)
plt.show()