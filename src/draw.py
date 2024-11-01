import matplotlib.pyplot as plt
import numpy as np
from main import determine_temperature
from config import EPOCHS, LR_START, LR_END

# Calculate temperature using determine_temperature
temperature = np.zeros(EPOCHS)
for i in range(EPOCHS):
    temperature[i] = determine_temperature(i)

# Calculate decay rate for ExponentialLR
decay_rate = (LR_END / LR_START) ** (1 / EPOCHS)

# Simulate learning rate schedule
learning_rates = [LR_START]
for _ in range(1, EPOCHS):
    learning_rates.append(learning_rates[-1] * decay_rate)

# Plotting temperature
plt.figure(figsize=(10, 6))
plt.plot(temperature, label='Temperature')
plt.title('Temperature Schedule Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.savefig('docs/temperature_schedule.png')

# Plotting learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, label='Learning Rate', color='orange')
plt.title('Learning Rate Schedule Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.legend()
plt.savefig('docs/learning_rate_schedule.png')


