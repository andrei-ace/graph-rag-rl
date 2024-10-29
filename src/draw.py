import matplotlib.pyplot as plt
import numpy as np
from main import determine_temperature
from config import EPOCHS

# Calculate temperature using determine_temperature
temperature = np.zeros(EPOCHS)
for i in range(EPOCHS):
    temperature[i] = determine_temperature(i)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(temperature, label='Temperature')
plt.title('Temperature Schedule Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.savefig('docs/temperature_schedule.png')
